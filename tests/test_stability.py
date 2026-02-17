import json
import os
import re
import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from poker_rl_agent.environment.openspiel_wrapper import PokerEnv
from poker_rl_agent.environment.state_representation import StateEncoder
from poker_rl_agent.evaluation.baseline_agents import AlwaysCallAgent, PassiveCallerAgent
from poker_rl_agent.evaluation.evaluator import Evaluator
from poker_rl_agent.models.alpha_holdem_net import AlphaHoldemNetwork
from poker_rl_agent.models.model_utils import masked_logits
from poker_rl_agent.training.action_curriculum import FullgameActionCurriculum
from poker_rl_agent.training.checkpointing import save_checkpoint
from poker_rl_agent.training.league_manager import LeagueManager
from poker_rl_agent.training.trainer import Trainer
from poker_rl_agent.utils import logging_utils
from poker_rl_agent.utils.config import Config
from poker_rl_agent.utils.logging_utils import log_metrics_local

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _decision_state(env):
    state = env.reset()
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        state.apply_action(outcomes[0][0])
    return state


def _test_config():
    cfg = Config()
    cfg.CFR_ITERATIONS = 1
    cfg.ROLLOUT_EPISODES = 4
    cfg.PPO_EPOCHS = 1
    cfg.PPO_MINIBATCH_SIZE = 8
    cfg.HIDDEN_DIM = 64
    cfg.NUM_LAYERS_CARD = 2
    cfg.NUM_LAYERS_ACTION = 1
    cfg.CHECKPOINT_FREQ = 1
    cfg.LOG_INTERVAL = 1
    cfg.EVAL_FREQ = 9999
    cfg.WANDB_MODE = "disabled"
    cfg.OFFLINE_LOGGING = False
    cfg.DEVICE = "cpu"
    cfg.MIN_ROLLOUT_TRANSITIONS = 4
    cfg.EVAL_BASELINE_ITERS = 2
    cfg.EVAL_CFR_ITERATIONS = 2
    cfg.EVAL_BASELINE_SEEDS = 1
    cfg.EVAL_BASELINE_CACHE_MODE = "process"
    cfg.RUN_ID = "test_run"
    cfg.PPO_ADV_CLIP = 5.0
    cfg.PPO_ADAPTIVE_KL_ENABLE = True
    cfg.PPO_ADAPTIVE_KL_HIGH = 2.0
    cfg.PPO_ADAPTIVE_KL_LOW = 0.5
    cfg.PPO_ADAPTIVE_LR_DECAY = 0.5
    cfg.PPO_ADAPTIVE_LR_GROWTH = 1.05
    cfg.EXPLAINED_VAR_VAR_FLOOR = 1e-4
    cfg.sync_legacy_fields()
    return cfg


def test_state_encoder_struct_features():
    env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
    state = _decision_state(env)

    encoder = StateEncoder(device="cpu", max_action_history=32)
    encoded = encoder.encode_state(state, state.current_player(), env.num_actions())

    assert encoded["hole_cards"].shape == (2,)
    assert encoded["community_cards"].shape == (5,)
    assert encoded["legal_action_mask"].shape == (env.num_actions(),)

    # Hole cards should be known after chance dealing in HUNL.
    assert (encoded["hole_cards"] != encoder.UNKNOWN_CARD).sum().item() >= 1
    # Pot/stack scalars should not be all zeros.
    assert torch.any(encoded["scalars"].abs() > 0)


def test_model_dual_head_output_and_masking():
    cfg = _test_config()
    env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
    state = _decision_state(env)

    encoder = StateEncoder(device="cpu", max_action_history=32)
    encoded = encoder.encode_state(state, state.current_player(), env.num_actions())
    batch = {k: v.unsqueeze(0) for k, v in encoded.items()}

    model = AlphaHoldemNetwork(env.num_actions(), cfg)
    out = model(batch)

    assert "policy_logits" in out and "state_value" in out
    assert out["policy_logits"].shape == (1, env.num_actions())
    assert out["state_value"].shape == (1,)

    masked = masked_logits(out["policy_logits"], batch["legal_action_mask"])
    illegal = (batch["legal_action_mask"] == 0)
    if illegal.any():
        assert torch.all(masked[illegal] < -1e8)


def test_solver_baseline_cache_key_and_wrapper():
    cfg = _test_config()
    env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
    model = AlphaHoldemNetwork(env.num_actions(), cfg)
    evaluator = Evaluator(model, config=cfg, device="cpu")

    stats_first = evaluator.evaluate_vs_solver_baseline(
        baseline_algo="mccfr_external_sampling",
        iterations=2,
        num_episodes=6,
        seed=123,
    )
    stats_cached = evaluator.evaluate_vs_solver_baseline(
        baseline_algo="mccfr_external_sampling",
        iterations=2,
        num_episodes=6,
        seed=123,
    )
    stats_new_seed = evaluator.evaluate_vs_solver_baseline(
        baseline_algo="mccfr_external_sampling",
        iterations=2,
        num_episodes=6,
        seed=124,
    )

    assert stats_first["baseline/algo"] == "mccfr_external_sampling"
    assert stats_first["baseline/cache_hit"] is False
    assert stats_cached["baseline/cache_hit"] is True
    assert stats_new_seed["baseline/cache_hit"] is False
    assert "baseline/build_seconds" in stats_first

    compat = evaluator.evaluate_vs_cfr(iterations=2, num_episodes=6, seed=125)
    assert compat["baseline/algo"] == "mccfr_external_sampling"


def test_solver_baseline_supports_zero_iterations_with_metadata():
    cfg = _test_config()
    env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
    model = AlphaHoldemNetwork(env.num_actions(), cfg)
    evaluator = Evaluator(model, config=cfg, device="cpu")

    zero_stats = evaluator.evaluate_vs_solver_baseline(
        baseline_algo="mccfr_external_sampling",
        iterations=0,
        num_episodes=4,
        seed=321,
        collect_diagnostics=True,
    )
    one_stats = evaluator.evaluate_vs_solver_baseline(
        baseline_algo="mccfr_external_sampling",
        iterations=1,
        num_episodes=4,
        seed=322,
        collect_diagnostics=True,
    )

    assert zero_stats["baseline/iters"] == 0
    assert one_stats["baseline/iters"] == 1
    assert zero_stats["baseline/build_seconds"] >= 0.0
    assert one_stats["baseline/build_seconds"] >= 0.0
    assert "diagnostics/action_freq_preflop/fold" in zero_stats


def test_solver_diagnostics_are_finite_and_bounded():
    cfg = _test_config()
    env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
    model = AlphaHoldemNetwork(env.num_actions(), cfg)
    evaluator = Evaluator(model, config=cfg, device="cpu")

    stats = evaluator.evaluate_vs_solver_baseline(
        baseline_algo="mccfr_external_sampling",
        iterations=1,
        num_episodes=8,
        seed=987,
        collect_diagnostics=True,
    )
    keys = [
        "diagnostics/action_freq_preflop/fold",
        "diagnostics/action_freq_preflop/call_check",
        "diagnostics/action_freq_preflop/half_pot",
        "diagnostics/action_freq_preflop/pot_raise",
        "diagnostics/action_freq_preflop/allin",
        "diagnostics/preflop_action_entropy_bits",
        "diagnostics/preflop_dominant_action_freq",
        "diagnostics/showdown_rate",
        "diagnostics/avg_pot_size_bb",
    ]
    for key in keys:
        assert np.isfinite(float(stats[key]))
    freq_sum = (
        float(stats["diagnostics/action_freq_preflop/fold"])
        + float(stats["diagnostics/action_freq_preflop/call_check"])
        + float(stats["diagnostics/action_freq_preflop/half_pot"])
        + float(stats["diagnostics/action_freq_preflop/pot_raise"])
        + float(stats["diagnostics/action_freq_preflop/allin"])
    )
    assert 0.0 <= freq_sum <= 1.0001
    assert 0.0 <= float(stats["diagnostics/preflop_dominant_action_freq"]) <= 1.0
    assert np.isfinite(float(stats["diagnostics/preflop_action_entropy_bits"]))
    assert 0.0 <= float(stats["diagnostics/showdown_rate"]) <= 1.0
    assert float(stats["diagnostics/avg_pot_size_bb"]) >= 0.0


def test_solver_diagnostics_include_legal_conditioned_rates():
    cfg = _test_config()
    cfg.ENV_PRESET = "hunl_fchpa"
    cfg.BETTING_ABSTRACTION = "fchpa"
    env = PokerEnv(env_preset="hunl_fchpa", betting_abstraction="fchpa")
    model = AlphaHoldemNetwork(env.num_actions(), cfg)
    evaluator = Evaluator(model, config=cfg, device="cpu")

    stats = evaluator.evaluate_vs_solver_baseline(
        baseline_algo="mccfr_external_sampling",
        iterations=1,
        num_episodes=8,
        seed=555,
        collect_diagnostics=True,
    )
    keys = [
        "diagnostics/preflop_legal_rate/pot_raise",
        "diagnostics/preflop_legal_rate/half_pot",
        "diagnostics/preflop_choose_given_legal/pot_raise",
        "diagnostics/preflop_choose_given_legal/half_pot",
    ]
    for key in keys:
        assert key in stats
        assert np.isfinite(float(stats[key]))
        assert 0.0 <= float(stats[key]) <= 1.0


def test_action_bucket_classification_supports_fchpa_alias():
    assert Evaluator._classify_action_bucket(0, "fchpa") == "fold"
    assert Evaluator._classify_action_bucket(1, "fchpa") == "call_check"
    assert Evaluator._classify_action_bucket(2, "fchpa") == "half_pot"
    assert Evaluator._classify_action_bucket(3, "fchpa") == "pot_raise"
    assert Evaluator._classify_action_bucket(4, "fchpa") == "allin"
    assert Evaluator._classify_action_bucket(0, "fcpha") == "fold"


def test_action_bucket_classification_supports_fullgame_strings():
    assert Evaluator._classify_action_bucket(10, "fullgame", action_text="player=0 move=Fold") == "fold"
    assert Evaluator._classify_action_bucket(11, "fullgame", action_text="player=0 move=Check") == "call_check"
    assert Evaluator._classify_action_bucket(12, "fullgame", action_text="player=0 move=Call") == "call_check"
    assert Evaluator._classify_action_bucket(13, "fullgame", action_text="player=0 move=AllIn") == "allin"
    assert (
        Evaluator._classify_action_bucket(
            14,
            "fullgame",
            action_text="player=0 move=RaiseTo 60",
            pot_size=100.0,
        )
        == "half_pot"
    )
    assert (
        Evaluator._classify_action_bucket(
            15,
            "fullgame",
            action_text="player=0 move=RaiseTo 300",
            pot_size=100.0,
        )
        == "pot_raise"
    )


def test_open_spiel_wrapper_normalizes_fcpha_to_fchpa():
    assert PokerEnv._normalize_betting_abstraction("fcpha") == "fchpa"
    assert PokerEnv._normalize_betting_abstraction("fchpa") == "fchpa"


def test_strict_abstraction_raises_when_requested_unavailable(monkeypatch):
    monkeypatch.setattr(
        PokerEnv,
        "_try_load_hunl_from_string",
        staticmethod(lambda abstraction: (None, f"{abstraction}:unsupported")),
    )
    monkeypatch.setattr(
        PokerEnv,
        "_try_load_fchpa_from_params",
        staticmethod(lambda: (None, "fchpa:param_unsupported")),
    )
    with pytest.raises(ValueError, match="STRICT_ABSTRACTION=true"):
        PokerEnv(
            game_name="universal_poker",
            env_preset="hunl_fchpa",
            betting_abstraction="fchpa",
            strict_abstraction=True,
        )


def test_fullgame_curriculum_mask_respects_phase_caps():
    class _Struct:
        pot_size = 100.0

    class _State:
        def is_terminal(self):
            return False

        def is_chance_node(self):
            return False

        def legal_actions(self):
            return [0, 1, 2, 3, 4]

        def action_to_string(self, _player, action):
            mapping = {
                0: "player=0 move=Fold",
                1: "player=0 move=Call",
                2: "player=0 move=RaiseTo 80",
                3: "player=0 move=RaiseTo 300",
                4: "player=0 move=AllIn",
            }
            return mapping[action]

        def to_struct(self):
            return _Struct()

    cfg = Config()
    cfg.FULLGAME_CURRICULUM_ENABLE = True
    cfg.FULLGAME_CURRICULUM_PHASE1_END = 0.30
    cfg.FULLGAME_CURRICULUM_PHASE2_END = 0.70
    cfg.FULLGAME_CURRICULUM_MAX_RAISE_POT_MULT_P1 = 1.0
    cfg.FULLGAME_CURRICULUM_MAX_RAISE_POT_MULT_P2 = 2.5

    curriculum = FullgameActionCurriculum(cfg)
    legal = torch.ones(5, dtype=torch.float32)
    state = _State()

    phase1 = curriculum.mask_for_state(state, player_id=0, num_actions=5, legal_action_mask=legal, progress=0.1)
    phase2 = curriculum.mask_for_state(state, player_id=0, num_actions=5, legal_action_mask=legal, progress=0.5)
    phase3 = curriculum.mask_for_state(state, player_id=0, num_actions=5, legal_action_mask=legal, progress=0.9)

    assert int(phase1[2].item()) == 1
    assert int(phase1[3].item()) == 0
    assert int(phase2[2].item()) == 1
    assert int(phase2[3].item()) == 0
    assert int(phase3[2].item()) == 1
    assert int(phase3[3].item()) == 1


def test_always_call_agent_prefers_non_fold_when_available():
    env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
    state = _decision_state(env)
    agent = AlwaysCallAgent()
    action = int(agent.step(state))
    assert action in state.legal_actions()
    action_name = state.action_to_string(state.current_player(), action).lower()
    if len(state.legal_actions()) > 1:
        assert "fold" not in action_name


def test_passive_caller_agent_avoids_raises_when_call_available():
    class _State:
        def legal_actions(self):
            return [0, 1, 2, 3]

        def current_player(self):
            return 0

        def action_to_string(self, _player, action):
            mapping = {
                0: "player=0 move=Fold",
                1: "player=0 move=Call",
                2: "player=0 move=Raise",
                3: "player=0 move=Bet",
            }
            return mapping[int(action)]

    agent = PassiveCallerAgent()
    action = int(agent.step(_State()))
    assert action == 1


def test_league_manager_capacity_and_sampling():
    cfg = _test_config()
    env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
    model = AlphaHoldemNetwork(env.num_actions(), cfg)

    league = LeagueManager(k_best=3, init_rating=1200.0, k_factor=24.0, pfsp_beta=2.0)
    for i in range(5):
        league.add_snapshot(model, step=i, score=float(i))

    assert len(league.entries) == 3
    sampled = league.sample_opponent()
    assert sampled is not None

    before = league.current_rating
    league.update_elo(sampled.entry_id, 1.0)
    assert league.current_rating != before


def test_short_training_smoke_run():
    cfg = _test_config()
    trainer = Trainer(cfg)
    trainer.train()

    assert trainer.global_step == cfg.CFR_ITERATIONS
    checkpoint_path = "checkpoints/point_1.pt"
    assert os.path.exists(checkpoint_path)

    for param in trainer.model.parameters():
        assert not torch.isnan(param).any()


def test_trainer_reports_stability_metrics():
    cfg = _test_config()
    trainer = Trainer(cfg)
    rollout_metrics = trainer._collect_rollouts()
    assert rollout_metrics["rollout/transitions"] >= 0
    for key in [
        "train/preflop_fold_freq",
        "train/preflop_call_check_freq",
        "train/preflop_half_pot_freq",
        "train/preflop_pot_raise_freq",
        "train/preflop_allin_freq",
    ]:
        assert key in rollout_metrics
        assert np.isfinite(float(rollout_metrics[key]))
    preflop_sum = (
        float(rollout_metrics["train/preflop_fold_freq"])
        + float(rollout_metrics["train/preflop_call_check_freq"])
        + float(rollout_metrics["train/preflop_half_pot_freq"])
        + float(rollout_metrics["train/preflop_pot_raise_freq"])
        + float(rollout_metrics["train/preflop_allin_freq"])
    )
    assert 0.0 <= preflop_sum <= 1.0001

    update_metrics = trainer._ppo_update()
    assert "train/clipfrac" in update_metrics
    assert "train/explained_var" in update_metrics
    assert "train/explained_var_raw" in update_metrics
    assert "train/value_target_var" in update_metrics
    assert "train/explained_var_valid_fraction" in update_metrics
    assert "train/kl_spike_events" in update_metrics
    assert "train/lr_multiplier" in update_metrics
    assert "train/nonfinite_grad_skips" in update_metrics
    assert "train/nonfinite_loss_batches" in update_metrics


def test_advantage_clipping_bounds_values_and_dtype():
    cfg = _test_config()
    trainer = Trainer(cfg)
    adv = torch.tensor([-9.0, -5.0, -1.0, 0.0, 2.0, 8.0], dtype=torch.float32)
    clipped = trainer._clip_advantages(adv)
    assert clipped.dtype == adv.dtype
    assert float(torch.max(clipped)) <= cfg.PPO_ADV_CLIP
    assert float(torch.min(clipped)) >= -cfg.PPO_ADV_CLIP


def test_temperature_schedule_is_bounded_and_monotonic():
    cfg = _test_config()
    cfg.SELF_PLAY_TEMPERATURE_START = 1.2
    cfg.SELF_PLAY_TEMPERATURE_END = 1.0
    cfg.SELF_PLAY_TEMPERATURE_DECAY_FRAC = 0.6
    trainer = Trainer(cfg)

    vals = [trainer._policy_temperature(x) for x in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    assert all(1.0 <= v <= 1.2 for v in vals)
    assert vals[0] >= vals[-1]
    assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))


def test_entropy_coef_schedule_is_bounded_and_monotonic():
    cfg = _test_config()
    cfg.PPO_ENTROPY_COEF_START = 0.010
    cfg.PPO_ENTROPY_COEF_END = 0.003
    cfg.PPO_ENTROPY_DECAY_FRAC = 0.7
    trainer = Trainer(cfg)

    vals = [trainer._entropy_coef(x) for x in [0.0, 0.2, 0.4, 0.7, 1.0]]
    assert all(0.003 <= v <= 0.010 for v in vals)
    assert vals[0] >= vals[-1]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))


def test_style_regularizer_disabled_forces_zero_coef():
    cfg = _test_config()
    cfg.STYLE_REG_ENABLE = False
    cfg.STYLE_REG_COEF_START = 0.02
    cfg.STYLE_REG_COEF_END = 0.005
    trainer = Trainer(cfg)
    assert trainer._style_reg_coef(0.0) == pytest.approx(0.0)
    assert trainer._style_reg_coef(1.0) == pytest.approx(0.0)


def test_schedule_progress_relative_to_resume_window():
    cfg = _test_config()
    cfg.CFR_ITERATIONS = 19000
    cfg.SCHEDULE_RELATIVE_TO_RESUME = True
    trainer = Trainer(cfg)

    start_step = 18000
    p_start = trainer._effective_schedule_progress(iteration=18000, start_step=start_step)
    p_mid = trainer._effective_schedule_progress(iteration=18499, start_step=start_step)
    p_end = trainer._effective_schedule_progress(iteration=18999, start_step=start_step)

    assert 0.0 < p_start < p_mid < p_end
    assert abs(p_start - (1.0 / 1000.0)) < 1e-9
    assert abs(p_mid - 0.5) < 1e-9
    assert abs(p_end - 1.0) < 1e-9


def test_adaptive_kl_controller_adjusts_learning_rate():
    cfg = _test_config()
    trainer = Trainer(cfg)
    initial_lr = trainer.optimizer.param_groups[0]["lr"]

    trainer._maybe_adjust_adaptive_lr(avg_kl=cfg.PPO_TARGET_KL * 3.0)
    lr_after_high = trainer.optimizer.param_groups[0]["lr"]
    assert lr_after_high < initial_lr

    for _ in range(10):
        trainer._maybe_adjust_adaptive_lr(avg_kl=cfg.PPO_TARGET_KL * 0.1)
    lr_after_low_streak = trainer.optimizer.param_groups[0]["lr"]
    assert lr_after_low_streak > lr_after_high


def test_trainer_resolves_opponent_mix_with_legacy_fallback():
    cfg = _test_config()
    cfg.LEAGUE_RANDOM_OPPONENT_PROB = 0.25
    cfg.LEAGUE_OPPONENT_PROB = 0.0
    cfg.EXPLOIT_OPPONENT_PROB = 0.0
    cfg.RANDOM_OPPONENT_PROB = 0.0
    trainer = Trainer(cfg)
    mix = trainer._resolve_opponent_mix()
    assert abs(mix["random"] - 0.25) < 1e-9
    assert abs(mix["league"] - 0.75) < 1e-9
    assert abs(mix["exploit"] - 0.0) < 1e-9


def test_trainer_resolves_opponent_mix_explicit_probs():
    cfg = _test_config()
    cfg.LEAGUE_OPPONENT_PROB = 0.60
    cfg.EXPLOIT_OPPONENT_PROB = 0.25
    cfg.RANDOM_OPPONENT_PROB = 0.15
    trainer = Trainer(cfg)
    mix = trainer._resolve_opponent_mix()
    assert abs(mix["league"] - 0.60) < 1e-6
    assert abs(mix["exploit"] - 0.25) < 1e-6
    assert abs(mix["random"] - 0.15) < 1e-6


def test_explained_var_uses_variance_floor_for_validity():
    cfg = _test_config()
    trainer = Trainer(cfg)
    value_pred = torch.zeros(8, dtype=torch.float32)
    # Constant targets -> near-zero variance should mark explained-var invalid.
    target = torch.ones(8, dtype=torch.float32) * 0.5
    raw, valid, var_y = trainer._compute_explained_var(value_pred, target)
    assert isinstance(raw, float)
    assert valid is False
    assert var_y < cfg.EXPLAINED_VAR_VAR_FLOOR


def test_masked_logits_fp16_safe():
    logits = torch.tensor([[0.1, -0.2, 0.3, 0.4]], dtype=torch.float16)
    legal = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float16)
    masked = masked_logits(logits, legal)

    # Should not overflow and illegal actions should be very negative.
    assert not torch.isnan(masked).any()
    assert masked.dtype == torch.float16
    assert masked[0, 1] < -1000
    assert masked[0, 3] < -1000


def test_minimal_single_file_checkpoint_mode():
    # Clean test artifacts from previous runs.
    os.makedirs("checkpoints", exist_ok=True)
    for file_name in os.listdir("checkpoints"):
        if file_name.endswith(".pt"):
            os.remove(os.path.join("checkpoints", file_name))

    cfg = _test_config()
    cfg.CFR_ITERATIONS = 2
    cfg.CHECKPOINT_FREQ = 1
    cfg.CHECKPOINT_SINGLE_FILE = True
    cfg.MAX_CHECKPOINTS = 1
    cfg.CHECKPOINT_SAVE_OPTIMIZER = False
    cfg.CHECKPOINT_SAVE_SCHEDULER = False
    cfg.CHECKPOINT_SAVE_LEAGUE = False
    cfg.CHECKPOINT_SAVE_EXTRA = False

    trainer = Trainer(cfg)
    trainer.train()

    checkpoint_files = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
    assert checkpoint_files == ["latest.pt"]

    payload = torch.load(os.path.join("checkpoints", "latest.pt"), map_location="cpu")
    assert payload.get("optimizer_state_dict") is None
    assert payload.get("scheduler_state_dict") is None
    assert payload.get("league_state") == {}


def test_single_file_checkpoint_keeps_protected_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        ckpt_dir = Path(temp_dir)
        latest = ckpt_dir / "latest.pt"
        best = ckpt_dir / "best_eval.pt"
        old = ckpt_dir / "old.pt"

        model = torch.nn.Linear(4, 2)
        optim_ = torch.optim.Adam(model.parameters(), lr=1e-3)
        torch.save({"x": 1}, best)
        torch.save({"x": 2}, old)

        save_checkpoint(
            model=model,
            optimizer=optim_,
            step=10,
            scheduler=None,
            path=str(latest),
            max_keep=1,
            single_file=True,
            save_optimizer=False,
            save_scheduler=False,
            save_league=False,
            save_extra=False,
            keep_files=[str(best)],
        )

        assert latest.exists()
        assert best.exists()
        assert not old.exists()


def test_trainer_best_eval_checkpoint_updates_only_on_improvement():
    os.makedirs("checkpoints", exist_ok=True)
    best_path = os.path.join("checkpoints", "best_eval_test.pt")
    if os.path.exists(best_path):
        os.remove(best_path)

    cfg = _test_config()
    cfg.CHECKPOINT_SINGLE_FILE = True
    cfg.CHECKPOINT_SAVE_BEST_EVAL = True
    cfg.CHECKPOINT_BEST_EVAL_PATH = best_path
    trainer = Trainer(cfg)

    out = trainer._maybe_save_best_eval(10, {"eval/mccfr_es_bb100": 123.0})
    assert out == best_path
    payload = torch.load(best_path, map_location="cpu")
    assert payload["extra"]["best_eval_score"] == 123.0
    assert payload["extra"]["best_eval_step"] == 10

    out2 = trainer._maybe_save_best_eval(20, {"eval/mccfr_es_bb100": 100.0})
    assert out2 is None
    payload2 = torch.load(best_path, map_location="cpu")
    assert payload2["extra"]["best_eval_score"] == 123.0
    assert payload2["extra"]["best_eval_step"] == 10


def test_log_metrics_local_uses_run_scoped_filename():
    with tempfile.TemporaryDirectory() as temp_dir:
        log_metrics_local({"run_id": "run-42", "x": 1}, step=1, out_dir=temp_dir)
        files = os.listdir(temp_dir)
        assert "metrics_run-42.jsonl" in files
        assert "metrics.jsonl" not in files


def test_wandb_preflight_requires_api_key_when_online(monkeypatch):
    cfg = Config()
    cfg.WANDB_MODE = "online"
    cfg.WANDB_REQUIRE_ONLINE = True
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        logging_utils.init_wandb(config=cfg)


def test_wandb_preflight_online_with_key_calls_init(monkeypatch):
    cfg = Config()
    cfg.WANDB_MODE = "online"
    cfg.WANDB_REQUIRE_ONLINE = True
    cfg.WANDB_PROJECT = "test_project"
    cfg.WANDB_ENTITY = "test_entity"
    cfg.WANDB_TAGS = "tag1,tag2"

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_TAGS", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    called = {}

    def _fake_init(**kwargs):
        called.update(kwargs)
        return object()

    monkeypatch.setattr(logging_utils.wandb, "init", _fake_init)
    run = logging_utils.init_wandb(config=cfg)
    assert run is not None
    assert called["mode"] == "online"
    assert called["project"] == "test_project"
    assert called["entity"] == "test_entity"
    assert called["tags"] == ["tag1", "tag2"]


def test_evaluate_script_writes_acceptance_fields():
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = Config()
        cfg.HIDDEN_DIM = 64
        cfg.NUM_LAYERS_CARD = 2
        cfg.NUM_LAYERS_ACTION = 1
        cfg.DEVICE = "cpu"
        env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
        model = AlphaHoldemNetwork(env.num_actions(), cfg)

        checkpoint_path = os.path.join(temp_dir, "tmp_eval_ckpt.pt")
        torch.save({"model_state_dict": model.state_dict(), "step": 0}, checkpoint_path)
        config_path = os.path.join(temp_dir, "eval_test_config.yaml")
        with open(config_path, "w", encoding="utf-8") as fh:
            fh.write(
                "default:\n"
                "  DEVICE: \"cpu\"\n"
                "  HIDDEN_DIM: 64\n"
                "  NUM_LAYERS_CARD: 2\n"
                "  NUM_LAYERS_ACTION: 1\n"
                "debug: {}\n"
            )

        output_json = os.path.join(temp_dir, "eval_out.json")
        script = str(PROJECT_ROOT / "poker_rl_agent" / "scripts" / "evaluate.py")
        cmd = [
            sys.executable,
            script,
            "--checkpoint",
            checkpoint_path,
            "--config_file",
            config_path,
            "--config_name",
            "debug",
            "--episodes_per_seed",
            "2",
            "--baseline_iters",
            "1",
            "--baseline_seeds",
            "1",
            "--output_json",
            output_json,
        ]
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        payload = json.loads(Path(output_json).read_text())
        assert "acceptance/ci95_lower_bb100" in payload
        assert "acceptance/ci95_upper_bb100" in payload
        assert "acceptance/pass_primary_gate" in payload
        assert isinstance(payload["acceptance/pass_primary_gate"], bool)


def test_stage_b_eval_slurm_script_is_present_and_valid_bash():
    slurm_path = str(PROJECT_ROOT / "scripts" / "slurm_stage_b_eval.slurm")
    assert os.path.exists(slurm_path)
    subprocess.run(["bash", "-n", slurm_path], check=True)


def test_stage_c_slurm_scripts_are_present_and_valid_bash():
    for name in ["slurm_stage_c_train.slurm", "slurm_stage_c_eval.slurm"]:
        slurm_path = str(PROJECT_ROOT / "scripts" / name)
        assert os.path.exists(slurm_path)
        subprocess.run(["bash", "-n", slurm_path], check=True)


def test_stage_d_slurm_scripts_are_present_and_valid_bash():
    for name in [
        "slurm_stage_d_fchpa_train.slurm",
        "slurm_stage_d_fchpa_eval.slurm",
        "slurm_stage_d_fcpha_train.slurm",
        "slurm_stage_d_fcpha_eval.slurm",
    ]:
        slurm_path = str(PROJECT_ROOT / "scripts" / name)
        assert os.path.exists(slurm_path)
        subprocess.run(["bash", "-n", slurm_path], check=True)


def test_stage_d_ablation_and_selection_slurm_scripts_are_present_and_valid_bash():
    for name in [
        "slurm_stage_d_fchpa_ablation_train.slurm",
        "slurm_stage_d_fchpa_ablation_eval.slurm",
        "slurm_stage_d_fchpa_selected_16k_train.slurm",
        "slurm_stage_d_fchpa_selected_16k_cert_eval.slurm",
        "slurm_stage_d_build_style_target.slurm",
    ]:
        slurm_path = str(PROJECT_ROOT / "scripts" / name)
        assert os.path.exists(slurm_path)
        subprocess.run(["bash", "-n", slurm_path], check=True)


def test_stage_e_slurm_scripts_are_present_and_valid_bash():
    for name in ["slurm_stage_e_fullgame_train.slurm", "slurm_stage_e_fullgame_eval.slurm"]:
        slurm_path = str(PROJECT_ROOT / "scripts" / name)
        assert os.path.exists(slurm_path)
        subprocess.run(["bash", "-n", slurm_path], check=True)


def test_stage_d_continue_preset_exists_and_resumes():
    cfg_path = PROJECT_ROOT / "configs" / "training_configs.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert "quadro_stage_d_fchpa_continue" in data
    preset = data["quadro_stage_d_fchpa_continue"]
    assert preset["BETTING_ABSTRACTION"] == "fchpa"
    assert preset["STRICT_ABSTRACTION"] is True
    assert preset["RESUME_FROM"] == "checkpoints/latest.pt"


def test_stage_d_ablation_presets_exist():
    cfg_path = PROJECT_ROOT / "configs" / "training_configs.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    for name in [
        "quadro_stage_d_fchpa_recover_explore_ablate",
        "quadro_stage_d_fchpa_recover_diverse_ablate",
        "quadro_stage_d_fchpa_recover_value_stable_ablate",
        "quadro_stage_d_fchpa_recover_explore_16k",
        "quadro_stage_d_fchpa_recover_diverse_16k",
        "quadro_stage_d_fchpa_recover_value_stable_16k",
        "quadro_stage_d_fchpa_recover_explore_corrective_17k",
        "quadro_stage_d_fchpa_recover_diverse_corrective_17k",
        "quadro_stage_d_fchpa_recover_value_stable_corrective_17k",
    ]:
        assert name in data
        assert data[name]["BETTING_ABSTRACTION"] == "fchpa"
        assert data[name]["STRICT_ABSTRACTION"] is True
        assert data[name]["RESUME_FROM"] == "checkpoints/latest.pt"


def test_stage_d_recovery_presets_exist():
    cfg_path = PROJECT_ROOT / "configs" / "training_configs.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    for name in [
        "quadro_stage_d_fchpa_recover_balanced_probe_19k",
        "quadro_stage_d_fchpa_recover_conservative_probe_19k",
        "quadro_stage_d_fchpa_recover_selected_20k",
        "quadro_stage_d_fchpa_recover_selected_20k_conservative",
        "quadro_stage_d_fchpa_recover_selected_20k_balanced",
        "quadro_stage_d_fchpa_recover_c1_strict_conservative_20k",
        "quadro_stage_d_fchpa_recover_c2_halfpot_nudge_20k",
    ]:
        assert name in data
        assert data[name]["BETTING_ABSTRACTION"] == "fchpa"
        assert data[name]["STRICT_ABSTRACTION"] is True
        assert "RESUME_FROM" in data[name]


def _write_temp_config(config_path: str):
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write(
            "default:\n"
            "  DEVICE: \"cpu\"\n"
            "  HIDDEN_DIM: 64\n"
            "  NUM_LAYERS_CARD: 2\n"
            "  NUM_LAYERS_ACTION: 1\n"
            "debug: {}\n"
        )


def test_play_against_agent_script_smoke_quit():
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = Config()
        cfg.HIDDEN_DIM = 64
        cfg.NUM_LAYERS_CARD = 2
        cfg.NUM_LAYERS_ACTION = 1
        cfg.DEVICE = "cpu"
        env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
        model = AlphaHoldemNetwork(env.num_actions(), cfg)

        checkpoint_path = os.path.join(temp_dir, "tmp_play_ckpt.pt")
        torch.save({"model_state_dict": model.state_dict(), "step": 0}, checkpoint_path)
        config_path = os.path.join(temp_dir, "play_test_config.yaml")
        _write_temp_config(config_path)

        script = str(PROJECT_ROOT / "poker_rl_agent" / "scripts" / "play_against_agent.py")
        cmd = [
            sys.executable,
            script,
            "--checkpoint",
            checkpoint_path,
            "--config_file",
            config_path,
            "--config_name",
            "debug",
            "--game_mode",
            "fcpa",
            "--hands",
            "1",
            "--human_seat",
            "0",
            "--seed",
            "123",
        ]
        proc = subprocess.run(
            cmd,
            input="quit\n",
            text=True,
            capture_output=True,
            check=True,
            cwd=str(PROJECT_ROOT),
            timeout=120,
        )
        assert "AlphaHoldEm Heads-Up CLI" in proc.stdout
        assert "Final session summary" in proc.stdout


def test_play_against_agent_input_validation_reprompts():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "play_against_agent.py"
    spec = importlib.util.spec_from_file_location("play_against_agent_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
    state = _decision_state(env)
    legal = state.legal_actions()
    assert len(legal) > 0

    import builtins

    original_input = builtins.input
    prompts = iter(["abc", str(legal[0])])
    try:
        builtins.input = lambda _: next(prompts)
        action, wants_quit = module._choose_human_action(state, state.current_player())
    finally:
        builtins.input = original_input

    assert wants_quit is False
    assert action in legal


def test_play_against_agent_fcpa_adapter_mapping():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "play_against_agent.py"
    spec = importlib.util.spec_from_file_location("play_against_agent_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class DummyState:
        def __init__(self):
            self._legal = [10, 11, 12, 13, 14]
            self._names = {
                10: "fold",
                11: "check",
                12: "raise 200",
                13: "raise 20000 all-in",
                14: "raise 1000",
            }

        def legal_actions(self):
            return list(self._legal)

        def action_to_string(self, _player, action):
            return self._names[action]

    state = DummyState()
    assert module._map_fcpa_choice_to_fullgame(state, 0, 0) == 10
    assert module._map_fcpa_choice_to_fullgame(state, 0, 1) == 11
    assert module._map_fcpa_choice_to_fullgame(state, 0, 2) in state.legal_actions()
    assert module._map_fcpa_choice_to_fullgame(state, 0, 3) == 13


def test_play_against_agent_disable_adapter_rejects_mismatch():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "play_against_agent.py"
    spec = importlib.util.spec_from_file_location("play_against_agent_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with pytest.raises(RuntimeError):
        module._resolve_action_space_config(
            game_mode="fullgame",
            env_num_actions=20001,
            checkpoint_num_actions=4,
            disable_adapter=True,
        )

    resolved_actions, adapter_mode = module._resolve_action_space_config(
        game_mode="fullgame",
        env_num_actions=20001,
        checkpoint_num_actions=4,
        disable_adapter=False,
    )
    assert resolved_actions == 4
    assert adapter_mode is True


def test_policy_temperature_scaling_shape_and_dtype():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "play_against_agent.py"
    spec = importlib.util.spec_from_file_location("play_against_agent_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    scaled = module._temperature_scale_logits(logits, 2.0)
    assert scaled.shape == logits.shape
    assert scaled.dtype == logits.dtype
    assert torch.allclose(scaled, logits / 2.0)

    floored = module._temperature_scale_logits(logits, 0.0)
    assert floored.shape == logits.shape
    assert torch.isfinite(floored).all()


def test_evaluate_complete_script_writes_expected_schema():
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = Config()
        cfg.HIDDEN_DIM = 64
        cfg.NUM_LAYERS_CARD = 2
        cfg.NUM_LAYERS_ACTION = 1
        cfg.DEVICE = "cpu"
        env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
        model = AlphaHoldemNetwork(env.num_actions(), cfg)

        checkpoint_path = os.path.join(temp_dir, "tmp_eval_complete_ckpt.pt")
        torch.save({"model_state_dict": model.state_dict(), "step": 0}, checkpoint_path)
        config_path = os.path.join(temp_dir, "eval_complete_test_config.yaml")
        _write_temp_config(config_path)

        output_json = os.path.join(temp_dir, "eval_complete_out.json")
        script = str(PROJECT_ROOT / "poker_rl_agent" / "scripts" / "evaluate_complete.py")
        cmd = [
            sys.executable,
            script,
            "--checkpoint",
            checkpoint_path,
            "--config_file",
            config_path,
            "--config_name",
            "debug",
            "--episodes_per_seed",
            "2",
            "--profile",
            "quick",
            "--seed",
            "123",
            "--output_json",
            output_json,
        ]
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), timeout=180)
        payload = json.loads(Path(output_json).read_text())
        assert "summary/random" in payload
        assert "summary/always_call" in payload
        assert "solver_tiers" in payload
        assert isinstance(payload["solver_tiers"], list)
        assert len(payload["solver_tiers"]) >= 2
        assert "overall/pass_default_cfr_gate" in payload
        assert "overall/pass_robustness_gate" in payload
        assert "overall/pass_holdout_gate" in payload
        assert "overall/pass_behavior_gate" in payload
        assert "overall/pass_behavior_gate_extended" in payload
        assert "overall/require_behavior_extended" in payload
        assert "overall/pass_all" in payload
        assert "verification/solver_training_verified" in payload
        assert "verification/control_zero_iter" in payload
        assert "verification/control_delta_bb100" in payload
        assert "diagnostics/summary" in payload
        assert "holdout" in payload
        assert "diagnostics/action_freq_preflop/half_pot" in payload["diagnostics/summary"]["random"]
        assert "diagnostics/preflop_action_entropy_bits" in payload["diagnostics/summary"]["random"]
        assert "diagnostics/preflop_dominant_action_freq" in payload["diagnostics/summary"]["random"]
        assert "behavior_gate" in payload
        assert "observed" in payload["behavior_gate"]
        assert "behavior_gate_extended" in payload
        assert "observed_call_check_freq" in payload["behavior_gate_extended"]
        assert "observed_half_pot_freq" in payload["behavior_gate_extended"]
        primary_seeds = {row.get("seed", row.get("baseline/seed")) for row in payload["solver_tiers"][0]["per_seed"]}
        holdout_seeds = {row["baseline/seed"] for row in payload["holdout"]["per_seed"]}
        assert primary_seeds.isdisjoint(holdout_seeds)
        tier0 = payload["solver_tiers"][0]
        for key in ["iterations", "seeds", "per_seed", "aggregate", "ci95_lower_bb100", "pass_gate"]:
            assert key in tier0


def test_evaluate_complete_reproducibility_same_seed():
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = Config()
        cfg.HIDDEN_DIM = 64
        cfg.NUM_LAYERS_CARD = 2
        cfg.NUM_LAYERS_ACTION = 1
        cfg.DEVICE = "cpu"
        env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
        model = AlphaHoldemNetwork(env.num_actions(), cfg)

        checkpoint_path = os.path.join(temp_dir, "tmp_eval_complete_ckpt.pt")
        torch.save({"model_state_dict": model.state_dict(), "step": 0}, checkpoint_path)
        config_path = os.path.join(temp_dir, "eval_complete_test_config.yaml")
        _write_temp_config(config_path)

        script = str(PROJECT_ROOT / "poker_rl_agent" / "scripts" / "evaluate_complete.py")
        out_a = os.path.join(temp_dir, "eval_complete_a.json")
        out_b = os.path.join(temp_dir, "eval_complete_b.json")
        base_cmd = [
            sys.executable,
            script,
            "--checkpoint",
            checkpoint_path,
            "--config_file",
            config_path,
            "--config_name",
            "debug",
            "--episodes_per_seed",
            "2",
            "--profile",
            "quick",
            "--seed",
            "777",
        ]

        subprocess.run(base_cmd + ["--output_json", out_a], check=True, cwd=str(PROJECT_ROOT), timeout=180)
        subprocess.run(base_cmd + ["--output_json", out_b], check=True, cwd=str(PROJECT_ROOT), timeout=180)

        a = json.loads(Path(out_a).read_text())
        b = json.loads(Path(out_b).read_text())
        assert len(a["solver_tiers"]) == len(b["solver_tiers"])
        for idx in range(len(a["solver_tiers"])):
            mean_a = a["solver_tiers"][idx]["aggregate"]["bb_per_100"]["mean"]
            mean_b = b["solver_tiers"][idx]["aggregate"]["bb_per_100"]["mean"]
            assert abs(mean_a - mean_b) < 1e-9


def test_evaluate_complete_respects_behavior_extended_requirement_flag():
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = Config()
        cfg.HIDDEN_DIM = 64
        cfg.NUM_LAYERS_CARD = 2
        cfg.NUM_LAYERS_ACTION = 1
        cfg.DEVICE = "cpu"
        env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
        model = AlphaHoldemNetwork(env.num_actions(), cfg)

        checkpoint_path = os.path.join(temp_dir, "tmp_eval_complete_ckpt.pt")
        torch.save({"model_state_dict": model.state_dict(), "step": 0}, checkpoint_path)
        config_path = os.path.join(temp_dir, "eval_complete_test_config.yaml")
        with open(config_path, "w", encoding="utf-8") as fh:
            fh.write(
                "default:\n"
                "  DEVICE: \"cpu\"\n"
                "  HIDDEN_DIM: 64\n"
                "  NUM_LAYERS_CARD: 2\n"
                "  NUM_LAYERS_ACTION: 1\n"
                "  EVAL_REQUIRE_BEHAVIOR_EXTENDED: false\n"
                "debug: {}\n"
            )

        script = str(PROJECT_ROOT / "poker_rl_agent" / "scripts" / "evaluate_complete.py")
        out_default = os.path.join(temp_dir, "eval_default.json")
        out_forced = os.path.join(temp_dir, "eval_forced.json")

        base_cmd = [
            sys.executable,
            script,
            "--checkpoint",
            checkpoint_path,
            "--config_file",
            config_path,
            "--config_name",
            "debug",
            "--episodes_per_seed",
            "2",
            "--profile",
            "quick",
            "--seed",
            "123",
        ]

        subprocess.run(base_cmd + ["--output_json", out_default], check=True, cwd=str(PROJECT_ROOT), timeout=180)
        subprocess.run(
            base_cmd + ["--require_behavior_extended", "--output_json", out_forced],
            check=True,
            cwd=str(PROJECT_ROOT),
            timeout=180,
        )

        default_payload = json.loads(Path(out_default).read_text())
        forced_payload = json.loads(Path(out_forced).read_text())
        assert default_payload["overall/require_behavior_extended"] is False
        assert forced_payload["overall/require_behavior_extended"] is True


def test_evaluate_complete_standard_profile_tiers():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "evaluate_complete.py"
    spec = importlib.util.spec_from_file_location("evaluate_complete_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    tiers = module.build_profile_tiers("standard", 5000, "fcpa")
    assert len(tiers) == 2
    assert tiers[0]["iterations"] == 5000
    assert tiers[0]["seeds"] == 5
    assert tiers[1]["iterations"] == 10000
    assert tiers[1]["seeds"] == 3


def test_evaluate_complete_standard_profile_tiers_fullgame():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "evaluate_complete.py"
    spec = importlib.util.spec_from_file_location("evaluate_complete_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    tiers = module.build_profile_tiers("standard", 5000, "fullgame")
    assert len(tiers) == 2
    assert tiers[0]["iterations"] == 2000
    assert tiers[0]["seeds"] == 5
    assert tiers[1]["iterations"] == 5000
    assert tiers[1]["seeds"] == 3


def test_evaluate_complete_verification_fails_for_zero_iter_tier():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "evaluate_complete.py"
    spec = importlib.util.spec_from_file_location("evaluate_complete_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    tiers = [
        {
            "name": "default_cfr",
            "iterations": 0,
            "per_seed": [{"baseline/build_seconds": 0.1}],
            "aggregate": {"bb_per_100": {"mean": 1.0}},
        }
    ]
    verified, delta, errors = module._compute_solver_verification(tiers, control_bb100=0.0)
    assert verified is False
    assert "default_cfr:iterations<=0" in errors
    assert "default_cfr" in delta


def test_evaluate_complete_aggression_gate_helper():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "evaluate_complete.py"
    spec = importlib.util.spec_from_file_location("evaluate_complete_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cfg = Config()
    cfg.EVAL_AGGRESSION_GATE_ENABLE = True
    cfg.EVAL_MIN_PRE_FLOP_RAISE_TOTAL_FREQ = 0.08

    metrics_row = {
        "diagnostics/action_freq_preflop/half_pot": 0.03,
        "diagnostics/action_freq_preflop/pot_raise": 0.03,
        "diagnostics/action_freq_preflop/allin": 0.03,
    }
    passed, details = module._passes_aggression_gate(cfg, metrics_row)
    assert passed is True
    assert details["observed_preflop_raise_total_freq"] == pytest.approx(0.09)

    metrics_row_low = {
        "diagnostics/action_freq_preflop/half_pot": 0.01,
        "diagnostics/action_freq_preflop/pot_raise": 0.01,
        "diagnostics/action_freq_preflop/allin": 0.01,
    }
    passed_low, details_low = module._passes_aggression_gate(cfg, metrics_row_low)
    assert passed_low is False
    assert details_low["min_preflop_raise_total_freq"] == pytest.approx(0.08)


def test_select_stage_d_winner_score_uses_exploit_bonus():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "select_stage_d_recovery_winner.py"
    spec = importlib.util.spec_from_file_location("select_stage_d_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as temp_dir:
        report_path = Path(temp_dir) / "candidate.json"
        payload = {
            "config_name": "cfg_probe",
            "checkpoint": "checkpoints/latest.pt",
            "verification/solver_training_verified": True,
            "overall/pass_default_cfr_gate": True,
            "overall/pass_robustness_gate": True,
            "overall/pass_holdout_gate": True,
            "overall/pass_behavior_gate": True,
            "overall/pass_behavior_gate_extended": True,
            "overall/pass_pot_mix_gate": True,
            "overall/exploit_weighted_bb100": 300.0,
            "solver_tiers": [
                {
                    "iterations": 5000,
                    "ci95_lower_bb100": 800.0,
                    "aggregate": {"bb_per_100": {"mean": 900.0}},
                    "diagnostics/action_freq_preflop/fold": 0.45,
                    "diagnostics/action_freq_preflop/call_check": 0.48,
                    "diagnostics/action_freq_preflop/half_pot": 0.06,
                    "diagnostics/action_freq_preflop/pot_raise": 0.01,
                    "diagnostics/preflop_choose_given_legal/pot_raise": 0.02,
                    "diagnostics/action_freq_preflop/allin": 0.01,
                    "diagnostics/preflop_action_entropy_bits": 1.3,
                },
                {
                    "iterations": 10000,
                    "ci95_lower_bb100": 700.0,
                    "aggregate": {"bb_per_100": {"mean": 850.0}},
                },
            ],
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")

        scored = module._score_candidate(
            name="probe_x",
            report_path=report_path,
            min_ci_floor=760.0,
            max_allin_freq=0.03,
            min_call_check_freq=0.47,
            min_half_pot_freq=0.01,
            min_raise_total_freq=0.079,
            min_pot_choose_given_legal_freq=0.01,
            min_fold_freq=0.41,
            max_fold_freq=0.52,
            min_entropy_bits=1.1,
            max_entropy_bits=1.8,
            exploit_weight=0.2,
            exploit_clip=500.0,
        )
        assert scored.eligible is True
        assert scored.exploit_clipped_bb100 == pytest.approx(300.0)
        assert scored.winner_score == pytest.approx(860.0)


def test_evaluate_complete_robustness_gate_toggle():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "evaluate_complete.py"
    spec = importlib.util.spec_from_file_location("evaluate_complete_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    tier = {
        "ci95_lower_bb100": -0.25,
        "aggregate": {"bb_per_100": {"mean": 0.1}},
    }
    assert module._passes_robustness_gate(tier, require_robust_ci=False) is True
    assert module._passes_robustness_gate(tier, require_robust_ci=True) is False


def test_evaluate_complete_behavior_gate_toggle():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "evaluate_complete.py"
    spec = importlib.util.spec_from_file_location("evaluate_complete_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cfg = Config()
    cfg.BEHAVIOR_GATE_ENABLE = True
    cfg.BEHAVIOR_GATE_MAX_FOLD_FREQ = 0.78
    cfg.BEHAVIOR_GATE_MAX_ALLIN_FREQ = 0.06
    cfg.BEHAVIOR_GATE_MIN_PRE_FLOP_ENTROPY_BITS = 0.75

    good = {
        "diagnostics/action_freq_preflop/fold": 0.70,
        "diagnostics/action_freq_preflop/allin": 0.03,
        "diagnostics/preflop_action_entropy_bits": 0.90,
    }
    bad_fold = {
        "diagnostics/action_freq_preflop/fold": 0.90,
        "diagnostics/action_freq_preflop/allin": 0.03,
        "diagnostics/preflop_action_entropy_bits": 0.90,
    }
    bad_entropy = {
        "diagnostics/action_freq_preflop/fold": 0.70,
        "diagnostics/action_freq_preflop/allin": 0.03,
        "diagnostics/preflop_action_entropy_bits": 0.30,
    }

    assert module._passes_behavior_gate(cfg, good) is True
    assert module._passes_behavior_gate(cfg, bad_fold) is False
    assert module._passes_behavior_gate(cfg, bad_entropy) is False


def test_evaluate_complete_extended_behavior_gate_toggle():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "evaluate_complete.py"
    spec = importlib.util.spec_from_file_location("evaluate_complete_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cfg = Config()
    cfg.BEHAVIOR_GATE_ENABLE = True
    cfg.BEHAVIOR_GATE_MIN_CALL_CHECK_FREQ = 0.50
    cfg.BEHAVIOR_GATE_MIN_HALF_POT_FREQ = 0.01

    good = {
        "diagnostics/action_freq_preflop/call_check": 0.60,
        "diagnostics/action_freq_preflop/half_pot": 0.02,
    }
    bad_call = {
        "diagnostics/action_freq_preflop/call_check": 0.40,
        "diagnostics/action_freq_preflop/half_pot": 0.02,
    }
    bad_half = {
        "diagnostics/action_freq_preflop/call_check": 0.60,
        "diagnostics/action_freq_preflop/half_pot": 0.0,
    }

    assert module._passes_behavior_gate_extended(cfg, good) is True
    assert module._passes_behavior_gate_extended(cfg, bad_call) is False
    assert module._passes_behavior_gate_extended(cfg, bad_half) is False


def test_evaluate_complete_primary_seed_set_helper():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "evaluate_complete.py"
    spec = importlib.util.spec_from_file_location("evaluate_complete_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    tiers = module.build_profile_tiers("standard", 10, "fcpa")
    seeds = module._primary_eval_seed_set(42, tiers)
    assert seeds == {42, 43, 44, 45, 46}


def test_wandb_import_payload_filtering():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "wandb_import_jsonl.py"
    spec = importlib.util.spec_from_file_location("wandb_import_jsonl_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    record = {
        "_step": 7,
        "train/policy_loss": -0.01,
        "train/value_loss": 0.23,
        "meta/config_name": "quadro_stage_c_fcpa",
        "flag": True,
    }
    step, payload = module._to_wandb_payload(record, "_step", re.compile(r"^train/"))
    assert step == 7
    assert "train/policy_loss" in payload
    assert "train/value_loss" in payload
    assert "meta/config_name" not in payload


def test_archive_run_script_creates_snapshot_manifest():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        checkpoint_path = temp / "latest.pt"
        metrics_path = temp / "metrics.jsonl"
        eval_path = temp / "eval.json"
        config_path = temp / "cfg.yaml"
        archive_root = temp / "artifacts"

        torch.save({"step": 123, "model_state_dict": {}}, checkpoint_path)
        metrics_path.write_text(
            json.dumps({"iteration": 120, "train/nonfinite_grad_skips": 0, "train/nonfinite_loss_batches": 0})
            + "\n",
            encoding="utf-8",
        )
        eval_path.write_text(
            json.dumps(
                {
                    "overall/pass_all": True,
                    "overall/pass_default_cfr_gate": True,
                    "overall/pass_robustness_gate": True,
                    "overall/pass_holdout_gate": True,
                    "verification/solver_training_verified": True,
                }
            ),
            encoding="utf-8",
        )
        config_path.write_text("default:\n  BETTING_ABSTRACTION: \"fcpa\"\n", encoding="utf-8")

        script = str(PROJECT_ROOT / "poker_rl_agent" / "scripts" / "archive_run.py")
        cmd = [
            sys.executable,
            script,
            "--run_name",
            "snapshot_test",
            "--stage",
            "stage_c_fcpa",
            "--checkpoint",
            str(checkpoint_path),
            "--metrics",
            str(metrics_path),
            "--eval_json",
            str(eval_path),
            "--config_file",
            str(config_path),
            "--config_name",
            "default",
            "--archive_root",
            str(archive_root),
        ]
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), timeout=120)
        run_dirs = [p for p in archive_root.iterdir() if p.is_dir()]
        assert len(run_dirs) == 1
        out = run_dirs[0]
        assert (out / "model" / "latest.pt").exists()
        assert (out / "eval" / "eval.json").exists()
        assert (out / "metrics" / "metrics.jsonl").exists()
        assert (out / "config" / "config_snapshot.yaml").exists()
        assert (out / "manifest.json").exists()
        assert checkpoint_path.exists()


def test_check_eval_gates_pass_and_fail():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "check_eval_gates.py"
    spec = importlib.util.spec_from_file_location("check_eval_gates_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    report = {
        "solver_tiers": [
            {
                "iterations": 5000,
                "aggregate": {"bb_per_100": {"mean": 900.0}},
                "ci95_lower_bb100": 840.0,
            },
            {
                "iterations": 10000,
                "aggregate": {"bb_per_100": {"mean": 910.0}},
                "ci95_lower_bb100": 850.0,
            },
        ],
        "overall/pass_default_cfr_gate": True,
        "overall/pass_robustness_gate": True,
        "overall/pass_holdout_gate": True,
        "overall/pass_behavior_gate": True,
        "overall/pass_behavior_gate_extended": True,
        "verification/solver_training_verified": True,
    }

    passed, summary = module.check_report(
        report,
        ci_floor=830.0,
        require_behavior_extended=True,
        require_solver_verified=True,
    )
    assert passed is True
    assert summary["checks"]["pass_ci_floor"] is True

    report["overall/pass_behavior_gate_extended"] = False
    passed2, summary2 = module.check_report(
        report,
        ci_floor=830.0,
        require_behavior_extended=True,
        require_solver_verified=True,
    )
    assert passed2 is False
    assert summary2["checks"]["pass_behavior_extended_gate"] is False


def test_check_eval_gates_behavior_envelope_thresholds():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "check_eval_gates.py"
    spec = importlib.util.spec_from_file_location("check_eval_gates_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    report = {
        "solver_tiers": [
            {
                "iterations": 5000,
                "aggregate": {"bb_per_100": {"mean": 880.0}},
                "ci95_lower_bb100": 780.0,
                "diagnostics/action_freq_preflop/fold": 0.48,
                "diagnostics/action_freq_preflop/call_check": 0.48,
                "diagnostics/action_freq_preflop/half_pot": 0.02,
                "diagnostics/action_freq_preflop/allin": 0.02,
                "diagnostics/preflop_action_entropy_bits": 1.3,
            },
            {
                "iterations": 10000,
                "aggregate": {"bb_per_100": {"mean": 860.0}},
                "ci95_lower_bb100": 760.0,
            },
        ],
        "overall/pass_default_cfr_gate": True,
        "overall/pass_robustness_gate": True,
        "overall/pass_holdout_gate": True,
        "overall/pass_behavior_gate": True,
        "overall/pass_behavior_gate_extended": True,
        "verification/solver_training_verified": True,
    }

    passed, summary = module.check_report(
        report,
        ci_floor=760.0,
        require_behavior_extended=True,
        require_solver_verified=True,
        max_allin_freq=0.03,
        min_call_check_freq=0.47,
        min_half_pot_freq=0.01,
        min_fold_freq=0.42,
        max_fold_freq=0.52,
        min_entropy_bits=1.1,
        max_entropy_bits=1.8,
    )
    assert passed is True
    assert summary["checks"]["pass_behavior_envelope"] is True

    report["solver_tiers"][0]["diagnostics/action_freq_preflop/allin"] = 0.08
    failed, summary_fail = module.check_report(
        report,
        ci_floor=760.0,
        require_behavior_extended=True,
        require_solver_verified=True,
        max_allin_freq=0.03,
        min_call_check_freq=0.47,
        min_half_pot_freq=0.01,
        min_fold_freq=0.42,
        max_fold_freq=0.52,
        min_entropy_bits=1.1,
        max_entropy_bits=1.8,
    )
    assert failed is False
    assert summary_fail["checks"]["pass_behavior_envelope"] is False


def test_stage_d_corrective_scripts_exist_and_are_shell_valid():
    script_paths = [
        PROJECT_ROOT / "scripts" / "slurm_stage_d_fchpa_corrective_train.slurm",
        PROJECT_ROOT / "scripts" / "slurm_stage_d_fchpa_corrective_screen_eval.slurm",
        PROJECT_ROOT / "scripts" / "slurm_stage_d_fchpa_corrective_cert_eval.slurm",
        PROJECT_ROOT / "scripts" / "slurm_stage_d_fchpa_eval_gate_check.slurm",
    ]
    for path in script_paths:
        assert path.exists(), f"Missing script: {path}"
        subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(PROJECT_ROOT), timeout=30)


def test_stage_d_recovery_scripts_exist_and_are_shell_valid():
    script_paths = [
        PROJECT_ROOT / "scripts" / "slurm_stage_d_fchpa_recovery_select.slurm",
        PROJECT_ROOT / "scripts" / "slurm_stage_d_fchpa_recovery_prepare_winner.slurm",
        PROJECT_ROOT / "scripts" / "submit_stage_d_fchpa_recovery_cycle.sh",
        PROJECT_ROOT / "scripts" / "run_stage_d_human_validation.sh",
    ]
    for path in script_paths:
        assert path.exists(), f"Missing script: {path}"
        subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(PROJECT_ROOT), timeout=30)


def test_resolve_stage_d_selected_config_mapping():
    module_path = PROJECT_ROOT / "poker_rl_agent" / "scripts" / "resolve_stage_d_selected_config.py"
    spec = importlib.util.spec_from_file_location("resolve_stage_d_selected_config_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    conservative = "quadro_stage_d_fchpa_recover_selected_20k_conservative"
    balanced = "quadro_stage_d_fchpa_recover_selected_20k_balanced"

    assert module.resolve_selected_config("probe_b", conservative, balanced, conservative) == conservative
    assert module.resolve_selected_config("probe_a", conservative, balanced, conservative) == balanced
    assert module.resolve_selected_config("unexpected", conservative, balanced, conservative) == conservative
    assert (
        module.resolve_selected_config(
            "probe_a",
            winner_config_name="quadro_stage_d_fchpa_recover_c1_strict_conservative_20k",
            conservative_config=conservative,
            balanced_config=balanced,
            fallback_config=conservative,
        )
        == conservative
    )
