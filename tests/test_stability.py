import os

import torch

from poker_rl_agent.environment.openspiel_wrapper import PokerEnv
from poker_rl_agent.environment.state_representation import StateEncoder
from poker_rl_agent.models.alpha_holdem_net import AlphaHoldemNetwork
from poker_rl_agent.models.model_utils import masked_logits
from poker_rl_agent.training.league_manager import LeagueManager
from poker_rl_agent.training.trainer import Trainer
from poker_rl_agent.utils.config import Config


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
