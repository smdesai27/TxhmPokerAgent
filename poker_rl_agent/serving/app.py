import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from poker_rl_agent.environment.openspiel_wrapper import PokerEnv
from poker_rl_agent.environment.state_representation import StateEncoder
from poker_rl_agent.models.alpha_holdem_net import AlphaHoldemNetwork
from poker_rl_agent.models.model_utils import masked_logits  # noqa: F401
from poker_rl_agent.training.checkpointing import load_checkpoint
from poker_rl_agent.utils.config import Config

from .game_manager import GameManager
from .schemas import (
    ActionRequest,
    ActionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    HealthResponse,
    NewHandResponse,
)

# Module-level storage for CLI args, read by lifespan.
_cli_args: dict = {}

STATIC_DIR = Path(__file__).parent / "static"


def _load_config(config_file: str, config_name: str) -> Config:
    config = Config()
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        merged = {}
        # Check if YAML uses nested layout (default: {...}) or flat layout (KEY: val)
        has_nested = any(isinstance(v, dict) for v in data.values())
        if has_nested:
            if "default" in data and isinstance(data["default"], dict):
                merged.update(data["default"])
            if config_name != "default" and config_name in data and isinstance(data[config_name], dict):
                merged.update(data[config_name])
        else:
            # Flat config snapshot (e.g. from archive_run)
            merged.update(data)
        for key, value in merged.items():
            if hasattr(config, key):
                setattr(config, key, value)
    if hasattr(config, "sync_legacy_fields"):
        config.sync_legacy_fields()
    return config


def _infer_checkpoint_num_actions(checkpoint_path: str) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_state = payload.get("model_state_dict", {})
    policy_bias = model_state.get("policy_head.bias")
    if policy_bias is None:
        raise KeyError("Checkpoint is missing model_state_dict['policy_head.bias'].")
    return int(policy_bias.shape[0])


game_manager: GameManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global game_manager

    args = _cli_args
    checkpoint = args.get("checkpoint", "checkpoints/latest.pt")
    config_file = args.get("config_file", "configs/training_configs.yaml")
    config_name = args.get("config_name", "default")
    game_mode = args.get("game_mode", "fcpa")
    bot_policy = args.get("bot_policy", "sample")
    policy_temperature = float(args.get("policy_temperature", 1.0))

    config = _load_config(config_file, config_name)
    config.DEVICE = "cpu"

    if game_mode == "fullgame":
        env_preset = "hunl_fullgame"
        betting_abstraction = "fullgame"
    elif game_mode in {"fcpha", "fchpa"}:
        env_preset = "hunl_fchpa"
        betting_abstraction = "fchpa"
    else:
        env_preset = "hunl_fcpa"
        betting_abstraction = "fcpa"

    env = PokerEnv(
        game_name=config.GAME_NAME,
        env_preset=env_preset,
        betting_abstraction=betting_abstraction,
        strict_abstraction=bool(getattr(config, "STRICT_ABSTRACTION", True)),
    )
    effective_abstraction = env.get_effective_betting_abstraction()
    env_num_actions = env.num_actions()

    checkpoint_num_actions = _infer_checkpoint_num_actions(checkpoint)
    if checkpoint_num_actions != env_num_actions:
        raise RuntimeError(
            f"Checkpoint num_actions={checkpoint_num_actions} != env num_actions={env_num_actions}. "
            "Adapter mode is not supported in the web server."
        )

    device = torch.device("cpu")
    model = AlphaHoldemNetwork(checkpoint_num_actions, config).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()

    encoder = StateEncoder(device="cpu", max_action_history=config.MAX_ACTION_HISTORY)
    bb_size = float(getattr(config, "BB_SIZE", getattr(config, "BIG_BLIND", 100.0)))

    game_manager = GameManager(
        model=model,
        encoder=encoder,
        config=config,
        device=device,
        num_actions=checkpoint_num_actions,
        game_name=config.GAME_NAME,
        env_preset=env_preset,
        betting_abstraction=effective_abstraction,
        strict_abstraction=bool(getattr(config, "STRICT_ABSTRACTION", True)),
        bot_policy=bot_policy,
        policy_temperature=policy_temperature,
        bb_size=bb_size,
    )

    # Background cleanup task
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(60)
            game_manager.cleanup_expired()

    cleanup_task = asyncio.create_task(_cleanup_loop())

    print(f"Server ready | mode={game_mode} | num_actions={checkpoint_num_actions} | bot_policy={bot_policy}")
    yield

    cleanup_task.cancel()
    game_manager = None


app = FastAPI(title="AlphaHoldem Poker", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/v1/health")
def health():
    return HealthResponse(
        status="ok",
        model_loaded=game_manager is not None,
        active_sessions=game_manager.active_session_count if game_manager else 0,
    )


@app.post("/api/v1/session")
def create_session(req: CreateSessionRequest):
    if game_manager is None:
        raise HTTPException(503, "Model not loaded")
    try:
        gs, stats = game_manager.create_session(req.human_seat)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return CreateSessionResponse(game_state=gs, stats=stats)


@app.get("/api/v1/session/{session_id}")
def get_session(session_id: str):
    if game_manager is None:
        raise HTTPException(503, "Model not loaded")
    try:
        gs, stats = game_manager.get_state(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found")
    return CreateSessionResponse(game_state=gs, stats=stats)


@app.post("/api/v1/session/{session_id}/action")
def submit_action(session_id: str, req: ActionRequest):
    if game_manager is None:
        raise HTTPException(503, "Model not loaded")
    try:
        gs, stats = game_manager.apply_human_action(session_id, req.action_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return ActionResponse(game_state=gs, stats=stats)


@app.post("/api/v1/session/{session_id}/new_hand")
def new_hand(session_id: str):
    if game_manager is None:
        raise HTTPException(503, "Model not loaded")
    try:
        gs, stats = game_manager.new_hand(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return NewHandResponse(game_state=gs, stats=stats)


def main():
    parser = argparse.ArgumentParser(description="AlphaHoldem Poker Web Server")
    parser.add_argument("--checkpoint", type=str, default="/Users/sanildesai/Documents/CodeProjects/AntigravityPokerAgent/checkpoints/latest.pt")
    parser.add_argument("--config_file", type=str, default="configs/training_configs.yaml")
    parser.add_argument("--config_name", type=str, default="default")
    parser.add_argument("--game_mode", type=str, choices=["fullgame", "fcpa", "fcpha", "fchpa"], default="fcpa")
    parser.add_argument("--bot_policy", type=str, choices=["sample", "argmax"], default="sample")
    parser.add_argument("--policy_temperature", type=float, default=1.0)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found at '{args.checkpoint}'", file=sys.stderr)
        sys.exit(1)

    _cli_args.update(vars(args))

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
