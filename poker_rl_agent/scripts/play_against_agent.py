import argparse
import os
import random
import re
import sys

import numpy as np
import torch
import yaml
from torch.distributions import Categorical

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from poker_rl_agent.environment.openspiel_wrapper import PokerEnv
from poker_rl_agent.environment.state_representation import StateEncoder
from poker_rl_agent.models.alpha_holdem_net import AlphaHoldemNetwork
from poker_rl_agent.models.model_utils import masked_logits
from poker_rl_agent.training.checkpointing import load_checkpoint
from poker_rl_agent.utils.config import Config


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(config_file: str, config_name: str) -> Config:
    config = Config()
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        merged = {}
        if "default" in data and isinstance(data["default"], dict):
            merged.update(data["default"])
        if config_name != "default" and config_name in data and isinstance(data[config_name], dict):
            merged.update(data[config_name])
        for key, value in merged.items():
            if hasattr(config, key):
                setattr(config, key, value)
    if hasattr(config, "sync_legacy_fields"):
        config.sync_legacy_fields()
    return config


def _cards_from_compact(blob: str):
    if not blob:
        return []
    return [blob[i : i + 2] for i in range(0, len(blob), 2)]


def _fmt_cards(blob: str) -> str:
    cards = _cards_from_compact(blob)
    return " ".join(cards) if cards else "-"


def _safe_action_name(state, player, action) -> str:
    try:
        return state.action_to_string(player, action)
    except Exception:
        return str(action)


def _print_public_state(state):
    struct = state.to_struct()
    board = _fmt_cards(getattr(struct, "board_cards", ""))
    pot = float(getattr(struct, "pot_size", 0.0))
    contrib = list(getattr(struct, "player_contributions", [0.0, 0.0]))
    starts = list(getattr(struct, "starting_stacks", [20000.0, 20000.0]))
    p0_stack = max(0.0, float(starts[0]) - float(contrib[0])) if len(starts) > 0 and len(contrib) > 0 else 0.0
    p1_stack = max(0.0, float(starts[1]) - float(contrib[1])) if len(starts) > 1 and len(contrib) > 1 else 0.0
    print("-" * 56)
    print(f"Board: {board}")
    print(
        "Pot: {:.1f} | Contributions P0/P1: {:.1f}/{:.1f} | Stacks P0/P1: {:.1f}/{:.1f}".format(
            pot,
            float(contrib[0]) if len(contrib) > 0 else 0.0,
            float(contrib[1]) if len(contrib) > 1 else 0.0,
            p0_stack,
            p1_stack,
        )
    )
    print(f"Current player: P{state.current_player()}")
    print("-" * 56)


def _choose_human_action(state, human_seat: int):
    legal = state.legal_actions()
    print("Legal actions:")
    for a in legal:
        print(f"  {a}: {_safe_action_name(state, human_seat, a)}")

    while True:
        raw = input("Type action id, 'help', or 'quit': ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            return None, True
        if raw in {"h", "help", "?"}:
            print("Enter one of the legal action ids shown above.")
            print("Use 'quit' to exit the current session.")
            continue
        try:
            action = int(raw)
        except ValueError:
            print(f"Invalid input '{raw}'. Try again.")
            continue

        if action not in legal:
            print(f"Invalid action {action}. Choose one of: {legal}")
            continue
        return action, False


def _temperature_scale_logits(logits: torch.Tensor, policy_temperature: float) -> torch.Tensor:
    temp = max(float(policy_temperature), 1e-3)
    return logits / temp


@torch.no_grad()
def _choose_bot_action(
    model,
    encoder,
    state,
    player_id: int,
    num_actions: int,
    bot_policy: str,
    policy_temperature: float,
):
    encoded = encoder.encode_state(state, player_id, num_actions)
    batch = {k: v.unsqueeze(0) for k, v in encoded.items()}
    outputs = model(batch)
    logits = masked_logits(outputs["policy_logits"], batch["legal_action_mask"])
    logits = _temperature_scale_logits(logits, policy_temperature)
    if bot_policy == "argmax":
        return int(torch.argmax(logits, dim=-1).item())
    return int(Categorical(logits=logits).sample().item())


def _resolve_human_seat(arg_value: str) -> int:
    if arg_value == "random":
        return int(np.random.choice([0, 1]))
    return int(arg_value)


def _infer_checkpoint_num_actions(checkpoint_path: str) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_state = payload.get("model_state_dict", {})
    policy_bias = model_state.get("policy_head.bias")
    if policy_bias is None:
        raise KeyError("Checkpoint is missing model_state_dict['policy_head.bias'].")
    return int(policy_bias.shape[0])


def _resolve_action_space_config(
    game_mode: str,
    env_num_actions: int,
    checkpoint_num_actions: int,
    disable_adapter: bool,
):
    if checkpoint_num_actions == env_num_actions:
        return env_num_actions, False

    if game_mode == "fullgame" and checkpoint_num_actions == 4 and not disable_adapter:
        return checkpoint_num_actions, True

    raise RuntimeError(
        "Failed to load checkpoint into the selected game mode. "
        f"Mode='{game_mode}' has num_actions={env_num_actions}, "
        f"checkpoint expects num_actions={checkpoint_num_actions}. "
        "Adapter is only available for fullgame + 4-action FCPA checkpoints unless --disable_adapter is set."
    )


def _extract_amount(text: str) -> float:
    nums = re.findall(r"\d+", text)
    if not nums:
        return -1.0
    return float(nums[-1])


def _categorize_legal_actions(state, player_id: int):
    legal = state.legal_actions()
    buckets = {
        "fold": [],
        "call_check": [],
        "raise_bet": [],
        "allin": [],
        "other": [],
    }
    for action in legal:
        name = _safe_action_name(state, player_id, action).lower()
        if "fold" in name:
            buckets["fold"].append((action, name))
        elif "check" in name or "call" in name:
            buckets["call_check"].append((action, name))
        elif "all-in" in name or "all in" in name or "allin" in name:
            buckets["allin"].append((action, name))
        elif "raise" in name or "bet" in name:
            buckets["raise_bet"].append((action, name))
        else:
            buckets["other"].append((action, name))
    return buckets


def _is_preflop_state(state) -> bool:
    try:
        return len(getattr(state.to_struct(), "board_cards", "")) == 0
    except Exception:
        return True


def _bucket_action_for_abstraction(
    abstraction: str,
    action: int,
    action_name: str,
    fcpa_idx: int = None,
) -> str:
    abstraction = str(abstraction).lower()

    # Adapter-mode index is the most reliable bucket for mapped fullgame actions.
    if fcpa_idx is not None:
        if fcpa_idx == 0:
            return "fold"
        if fcpa_idx == 1:
            return "call_check"
        if fcpa_idx == 2:
            return "pot_raise"
        if fcpa_idx == 3:
            return "allin"
        return "other"

    if abstraction == "fcpa":
        if action == 0:
            return "fold"
        if action == 1:
            return "call_check"
        if action == 2:
            return "pot_raise"
        if action == 3:
            return "allin"
        return "other"

    if abstraction in {"fchpa", "fcpha"}:
        if action == 0:
            return "fold"
        if action == 1:
            return "call_check"
        if action == 2:
            return "half_pot"
        if action == 3:
            return "pot_raise"
        if action == 4:
            return "allin"
        return "other"

    lower = str(action_name).lower()
    if "fold" in lower:
        return "fold"
    if "call" in lower or "check" in lower:
        return "call_check"
    if "all-in" in lower or "all in" in lower or "allin" in lower:
        return "allin"
    if "raise" in lower or "bet" in lower:
        return "pot_raise"
    return "other"


def _map_fcpa_choice_to_fullgame(state, player_id: int, fcpa_idx: int) -> int:
    buckets = _categorize_legal_actions(state, player_id)
    legal = state.legal_actions()
    if not legal:
        raise RuntimeError("No legal actions available in non-terminal state.")

    # FCPA index convention: 0=fold, 1=call/check, 2=pot-ish raise, 3=all-in.
    if fcpa_idx == 0:
        if buckets["fold"]:
            return int(buckets["fold"][0][0])
        if buckets["call_check"]:
            return int(buckets["call_check"][0][0])
        return int(legal[0])

    if fcpa_idx == 1:
        if buckets["call_check"]:
            return int(buckets["call_check"][0][0])
        non_fold = [a for a, _ in (buckets["raise_bet"] + buckets["allin"] + buckets["other"])]
        if non_fold:
            return int(non_fold[0])
        return int(legal[0])

    if fcpa_idx == 2:
        candidates = buckets["raise_bet"] + buckets["allin"]
        if candidates:
            ordered = sorted(candidates, key=lambda x: (_extract_amount(x[1]), x[0]))
            mid = len(ordered) // 2
            return int(ordered[mid][0])
        if buckets["call_check"]:
            return int(buckets["call_check"][0][0])
        return int(legal[-1])

    # all-in branch (idx 3 or fallback)
    if buckets["allin"]:
        ordered = sorted(buckets["allin"], key=lambda x: (_extract_amount(x[1]), x[0]))
        return int(ordered[-1][0])
    candidates = buckets["raise_bet"]
    if candidates:
        ordered = sorted(candidates, key=lambda x: (_extract_amount(x[1]), x[0]))
        return int(ordered[-1][0])
    if buckets["call_check"]:
        return int(buckets["call_check"][0][0])
    return int(legal[-1])


@torch.no_grad()
def _choose_bot_action_fcpa_adapter(
    model,
    encoder,
    state,
    player_id: int,
    model_num_actions: int,
    bot_policy: str,
    policy_temperature: float,
):
    encoded = encoder.encode_state(state, player_id, model_num_actions)
    batch = {k: v.unsqueeze(0) for k, v in encoded.items()}
    # Adapter mode does not trust numeric id overlap from fullgame legal actions.
    batch["legal_action_mask"] = torch.ones((1, model_num_actions), dtype=torch.float32, device=batch["hole_cards"].device)
    outputs = model(batch)
    logits = _temperature_scale_logits(outputs["policy_logits"], policy_temperature)
    if bot_policy == "argmax":
        fcpa_idx = int(torch.argmax(logits, dim=-1).item())
    else:
        fcpa_idx = int(Categorical(logits=logits).sample().item())
    return _map_fcpa_choice_to_fullgame(state, player_id, fcpa_idx), fcpa_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--config_file", type=str, default="configs/training_configs.yaml")
    parser.add_argument("--config_name", type=str, default="quadro_stage_b")
    parser.add_argument("--game_mode", type=str, choices=["fullgame", "fcpa", "fcpha", "fchpa"], default="fullgame")
    parser.add_argument("--human_seat", type=str, choices=["0", "1", "random"], default="0")
    parser.add_argument("--hands", type=int, default=20)
    parser.add_argument("--bot_policy", type=str, choices=["sample", "argmax"], default="sample")
    parser.add_argument("--policy_temperature", type=float, default=1.0)
    parser.add_argument("--disable_adapter", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found at '{args.checkpoint}'. "
            "Pass --checkpoint with a valid .pt path."
        )

    config = _load_config(args.config_file, args.config_name)
    strict_abstraction_env = os.environ.get("STRICT_ABSTRACTION")
    if strict_abstraction_env is not None:
        config.STRICT_ABSTRACTION = str(strict_abstraction_env).strip().lower() in {"1", "true", "yes", "on"}
    if args.seed is not None:
        config.SEED = int(args.seed)
    if hasattr(config, "sync_legacy_fields"):
        config.sync_legacy_fields()

    if args.game_mode == "fullgame":
        env_preset = "hunl_fullgame"
        betting_abstraction = "fullgame"
    elif args.game_mode in {"fcpha", "fchpa"}:
        env_preset = "hunl_fchpa"
        betting_abstraction = "fchpa"
    else:
        env_preset = "hunl_fcpa"
        betting_abstraction = "fcpa"

    _set_seed(int(config.SEED))

    env = PokerEnv(
        game_name=config.GAME_NAME,
        env_preset=env_preset,
        betting_abstraction=betting_abstraction,
        strict_abstraction=bool(getattr(config, "STRICT_ABSTRACTION", True)),
    )
    effective_abstraction = env.get_effective_betting_abstraction()
    if effective_abstraction != betting_abstraction:
        print(
            "Requested game abstraction "
            f"'{betting_abstraction}' is not supported by this OpenSpiel build; "
            f"using '{effective_abstraction}' instead."
        )
    betting_abstraction = effective_abstraction
    env_num_actions = env.num_actions()
    checkpoint_num_actions = _infer_checkpoint_num_actions(args.checkpoint)

    model_num_actions, adapter_mode = _resolve_action_space_config(
        game_mode=args.game_mode,
        env_num_actions=env_num_actions,
        checkpoint_num_actions=checkpoint_num_actions,
        disable_adapter=bool(args.disable_adapter),
    )

    target_device = torch.device(config.DEVICE)
    if adapter_mode:
        print(
            "Checkpoint/action-space mismatch detected: "
            f"checkpoint_num_actions={checkpoint_num_actions}, env_num_actions={env_num_actions}. "
            "Using FCPA-to-fullgame adapter."
        )

    model = AlphaHoldemNetwork(model_num_actions, config).to(target_device)
    load_checkpoint(args.checkpoint, model, map_location=target_device)
    model.eval()

    encoder = StateEncoder(device=str(target_device), max_action_history=config.MAX_ACTION_HISTORY)

    bb_size = float(getattr(config, "BB_SIZE", getattr(config, "BIG_BLIND", 100.0)))
    total_human = 0.0
    total_bot = 0.0
    played_hands = 0
    bot_preflop_total = 0
    bot_preflop_counts = {
        "fold": 0,
        "call_check": 0,
        "half_pot": 0,
        "pot_raise": 0,
        "allin": 0,
        "other": 0,
    }

    print("AlphaHoldEm Heads-Up CLI")
    print(f"Checkpoint: {args.checkpoint}")
    print(
        f"Mode: {args.game_mode} | Bot policy: {args.bot_policy} | Device: {target_device} | "
        f"adapter_mode={adapter_mode} | policy_temperature={args.policy_temperature:.3f}"
    )
    if adapter_mode:
        print("Adapter mode active: qualitative play only, not benchmark-comparable.")
    print(f"Seed: {config.SEED} | Target hands: {args.hands}")

    for hand_idx in range(1, args.hands + 1):
        human_seat = _resolve_human_seat(args.human_seat)
        bot_seat = 1 - human_seat
        state = env.reset()

        print("\n" + "=" * 72)
        print(f"Hand {hand_idx}/{args.hands} | Human=P{human_seat} | Bot=P{bot_seat}")

        user_quit = False
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action_list, probs = zip(*outcomes)
                sampled = int(np.random.choice(action_list, p=probs))
                state.apply_action(sampled)
                continue

            current_player = state.current_player()
            _print_public_state(state)

            if current_player == human_seat:
                struct = state.to_struct()
                hands = list(getattr(struct, "player_hands", []))
                hero_cards = hands[human_seat] if human_seat < len(hands) else ""
                print(f"Your hole cards: {_fmt_cards(hero_cards)}")
                action, wants_quit = _choose_human_action(state, human_seat)
                if wants_quit:
                    user_quit = True
                    break
                state.apply_action(int(action))
            else:
                fcpa_idx = None
                if adapter_mode:
                    action, fcpa_idx = _choose_bot_action_fcpa_adapter(
                        model=model,
                        encoder=encoder,
                        state=state,
                        player_id=bot_seat,
                        model_num_actions=model_num_actions,
                        bot_policy=args.bot_policy,
                        policy_temperature=args.policy_temperature,
                    )
                else:
                    action = _choose_bot_action(
                        model=model,
                        encoder=encoder,
                        state=state,
                        player_id=bot_seat,
                        num_actions=model_num_actions,
                        bot_policy=args.bot_policy,
                        policy_temperature=args.policy_temperature,
                    )
                action_name = _safe_action_name(state, bot_seat, action)
                if _is_preflop_state(state):
                    bucket = _bucket_action_for_abstraction(
                        abstraction=betting_abstraction,
                        action=action,
                        action_name=action_name,
                        fcpa_idx=fcpa_idx,
                    )
                    bot_preflop_total += 1
                    bot_preflop_counts[bucket] = int(bot_preflop_counts.get(bucket, 0)) + 1
                if fcpa_idx is None:
                    print(f"Bot action: {action} ({action_name})")
                else:
                    print(f"Bot action: {action} ({action_name}) [fcpa_idx={fcpa_idx}]")
                state.apply_action(action)

        if user_quit:
            print("Session ended by user.")
            break

        returns = state.returns()
        human_return = float(returns[human_seat])
        bot_return = float(returns[bot_seat])
        total_human += human_return
        total_bot += bot_return
        played_hands += 1

        struct = state.to_struct()
        board_cards = _fmt_cards(getattr(struct, "board_cards", ""))
        player_hands = list(getattr(struct, "player_hands", []))
        p0_hole = _fmt_cards(player_hands[0]) if len(player_hands) > 0 else "-"
        p1_hole = _fmt_cards(player_hands[1]) if len(player_hands) > 1 else "-"

        print("Hand complete.")
        print(f"Board: {board_cards}")
        print(f"Hole cards P0/P1: {p0_hole} / {p1_hole}")
        print(
            "Hand return Human/Bot: {:.1f} / {:.1f} chips ({:.2f} / {:.2f} bb)".format(
                human_return,
                bot_return,
                human_return / bb_size,
                bot_return / bb_size,
            )
        )
        print(
            "Cumulative Human/Bot: {:.1f} / {:.1f} chips ({:.2f} / {:.2f} bb)".format(
                total_human,
                total_bot,
                total_human / bb_size,
                total_bot / bb_size,
            )
        )

    print("\nFinal session summary")
    print(f"Played hands: {played_hands}")
    print(
        "Total Human/Bot: {:.1f} / {:.1f} chips ({:.2f} / {:.2f} bb)".format(
            total_human,
            total_bot,
            total_human / bb_size if bb_size > 0 else 0.0,
            total_bot / bb_size if bb_size > 0 else 0.0,
        )
    )
    if bot_preflop_total > 0:
        print("Bot preflop action distribution:")
        for key in ["fold", "call_check", "half_pot", "pot_raise", "allin", "other"]:
            value = int(bot_preflop_counts.get(key, 0))
            freq = float(value) / float(bot_preflop_total)
            print(f"  {key}: {value}/{bot_preflop_total} ({freq:.4f})")


if __name__ == "__main__":
    main()
