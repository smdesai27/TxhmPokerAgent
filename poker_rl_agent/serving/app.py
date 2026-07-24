import argparse
import asyncio
import os
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

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

# Module-level storage for CLI args (read by lifespan) + server info (read by /health).
_cli_args: dict = {}
_server_info: dict = {}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Lightweight in-memory per-IP rate limiter (no external dependency).

    Protects the public demo from trivial abuse: a global request cap plus a
    stricter cap on session creation. Honors X-Forwarded-For (Render/Vercel proxy)
    so the limit keys on the real client, not the proxy. Only the /api/ surface is
    policed; static assets and the SPA are unrestricted.
    """

    def __init__(
        self,
        app,
        global_max: int = 90,
        global_window: float = 60.0,
        session_create_max: int = 15,
        session_create_window: float = 300.0,
    ):
        super().__init__(app)
        self._global = (int(global_max), float(global_window))
        self._session = (int(session_create_max), float(session_create_window))
        self._hits: dict = {}  # (ip, bucket) -> deque[timestamps]
        self._lock = threading.Lock()

    @staticmethod
    def _client_ip(request) -> str:
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _allow(self, key, limit: int, window: float, now: float):
        dq = self._hits.get(key)
        if dq is None:
            dq = deque()
            self._hits[key] = dq
        cutoff = now - window
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit:
            return False, int(dq[0] + window - now) + 1
        dq.append(now)
        return True, 0

    async def dispatch(self, request, call_next):
        path = request.url.path
        if path.startswith("/api/"):
            ip = self._client_ip(request)
            now = time.time()
            checks = [((ip, "global"), *self._global)]
            if request.method == "POST" and path.endswith("/api/v1/session"):
                checks.append(((ip, "session"), *self._session))
            with self._lock:
                if len(self._hits) > 10000:  # opportunistic prune to bound memory
                    for k in [k for k, d in self._hits.items() if not d]:
                        self._hits.pop(k, None)
                for key, limit, window in checks:
                    ok, retry = self._allow(key, limit, window, now)
                    if not ok:
                        return JSONResponse(
                            {"detail": "Rate limit exceeded. Please slow down."},
                            status_code=429,
                            headers={"Retry-After": str(retry)},
                        )
        return await call_next(request)

# Prefer project-root public/ (canonical source), fall back to bundled static/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = _PROJECT_ROOT / "public"
if not STATIC_DIR.is_dir():
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
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
    checkpoint = args.get("checkpoint") or os.environ.get(
        "CHECKPOINT_PATH", "checkpoints/snapshots/interview_ready/interview_ready_1.pt"
    )
    config_file = args.get("config_file") or os.environ.get("CONFIG_FILE", "configs/training_configs.yaml")
    config_name = args.get("config_name") or os.environ.get("CONFIG_NAME", "default")
    game_mode = args.get("game_mode") or os.environ.get("GAME_MODE", "fchpa")
    bot_policy = args.get("bot_policy") or os.environ.get("BOT_POLICY", "sample")
    policy_temperature = float(args.get("policy_temperature") or os.environ.get("POLICY_TEMPERATURE", "1.0"))

    config = _load_config(config_file, config_name)
    config.DEVICE = "cpu"

    # Free-tier boxes are single-vCPU with a 512MB cap. Limit torch's intra-op
    # thread pool so it doesn't spawn one worker per host core (inflating RSS and
    # thrashing the shared core). Safe no-op if already constrained via env.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

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

    _server_info.update({
        "checkpoint_label": os.environ.get("CHECKPOINT_LABEL", "") or Path(checkpoint).name,
        "game_mode": game_mode,
        "num_actions": checkpoint_num_actions,
    })
    print(f"Server ready | mode={game_mode} | num_actions={checkpoint_num_actions} | bot_policy={bot_policy}")
    yield

    cleanup_task.cancel()
    game_manager = None


app = FastAPI(title="AlphaHoldem Poker", lifespan=lifespan)

# Added before CORS so CORS (added below) wraps it and 429 responses still carry
# the CORS headers a browser needs to surface the error correctly.
app.add_middleware(
    RateLimitMiddleware,
    global_max=90,
    global_window=60.0,
    session_create_max=15,
    session_create_window=300.0,
)

cors_origins = os.environ.get("CORS_ORIGINS", "")
origins = cors_origins.split(",") if cors_origins else [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

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
        checkpoint_label=_server_info.get("checkpoint_label", ""),
        num_actions=game_manager.num_actions if game_manager else 0,
        game_mode=_server_info.get("game_mode", ""),
    )


@app.post("/api/v1/session")
def create_session(req: CreateSessionRequest):
    if game_manager is None:
        raise HTTPException(503, "Model not loaded")
    try:
        gs, stats = game_manager.create_session(req.human_seat)
    except ValueError as e:
        msg = str(e)
        if "Too many active sessions" in msg:
            raise HTTPException(429, msg)
        raise HTTPException(400, msg)
    return CreateSessionResponse(game_state=gs, stats=stats)


@app.get("/api/v1/session/{session_id}")
def get_session(session_id: str):
    if game_manager is None:
        raise HTTPException(503, "Model not loaded")
    try:
        gs, stats = game_manager.get_state(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found")
    return CreateSessionResponse(game_state=gs, stats=stats)


@app.post("/api/v1/session/{session_id}/action")
def submit_action(session_id: str, req: ActionRequest):
    if game_manager is None:
        raise HTTPException(503, "Model not loaded")
    try:
        gs, stats = game_manager.apply_human_action(session_id, req.action_id)
    except KeyError:
        raise HTTPException(404, "Session not found")
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
        raise HTTPException(404, "Session not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return NewHandResponse(game_state=gs, stats=stats)


# Serve the single-page frontend and its root-path assets (index.html, app.js,
# style.css, config.js). Mounted LAST so /api routes and /static take precedence;
# html=True serves index.html at "/". Fixes asset 404s when the backend (not
# Vercel) serves the page directly.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="spa")


def main():
    parser = argparse.ArgumentParser(description="AlphaHoldem Poker Web Server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/snapshots/interview_ready/interview_ready_1.pt",
    )
    parser.add_argument("--config_file", type=str, default="configs/training_configs.yaml")
    parser.add_argument("--config_name", type=str, default="default")
    parser.add_argument("--game_mode", type=str, choices=["fullgame", "fcpa", "fcpha", "fchpa"], default="fchpa")
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
