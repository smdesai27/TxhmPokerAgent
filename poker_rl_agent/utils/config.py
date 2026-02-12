import torch
from dataclasses import dataclass, field
from typing import Dict, Any


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # Environment / game settings
    GAME_NAME: str = "universal_poker"
    ENV_PRESET: str = "hunl_fcpa"
    BETTING_ABSTRACTION: str = "fcpa"  # fcpa | fullgame
    PLAYERS: int = 2
    STACK_SIZE: int = 20000
    SMALL_BLIND: int = 50
    BIG_BLIND: int = 100
    MAX_ACTION_HISTORY: int = 64
    NUM_ACTORS: int = 1

    # Model
    EMBEDDING_DIM: int = 64
    HIDDEN_DIM: int = 256
    NUM_LAYERS_CARD: int = 3
    NUM_LAYERS_ACTION: int = 2
    DROPOUT: float = 0.1
    USE_LAYERNORM: bool = True
    USE_RESIDUAL: bool = True
    SCALAR_DIM: int = 11

    # Algorithm
    ALGO: str = "ppo_ksp"
    CFR_ITERATIONS: int = 100000  # kept for backwards compatibility with scripts
    ROLLOUT_EPISODES: int = 64
    SELF_PLAY_GAMES: int = 20  # compatibility field for old configs/scripts
    PPO_EPOCHS: int = 4
    PPO_MINIBATCH_SIZE: int = 256
    PPO_CLIP_EPS: float = 0.2
    PPO_GAMMA: float = 0.995
    PPO_GAE_LAMBDA: float = 0.95
    PPO_VALUE_COEF: float = 0.5
    PPO_ENTROPY_COEF: float = 0.01
    PPO_TARGET_KL: float = 0.03
    VALUE_LOSS_TYPE: str = "huber"  # huber | mse
    VALUE_HUBER_DELTA: float = 10.0
    VALUE_TARGET_SCALE: float = 20.0
    SKIP_NONFINITE_GRAD: bool = True
    MIN_ROLLOUT_TRANSITIONS: int = 64

    # K-best league self-play
    K_BEST: int = 8
    LEAGUE_INIT_RATING: float = 1200.0
    LEAGUE_K_FACTOR: float = 24.0
    PFSP_BETA: float = 2.0
    SNAPSHOT_INTERVAL: int = 50

    # Optimization / runtime
    LR: float = 3e-4
    WEIGHT_DECAY: float = 1e-5
    BATCH_SIZE: int = 256
    BUFFER_SIZE: int = 500000  # compatibility field for old configs/scripts
    GRAD_CLIP: float = 1.0
    GRAD_ACCUM_STEPS: int = 1
    AMP: bool = True
    LR_SCHEDULER: str = "none"  # none | plateau

    # Evaluation
    EVAL_FREQ: int = 50
    EVAL_EPISODES: int = 200
    EVAL_BASELINE_ALGO: str = "mccfr_external_sampling"
    EVAL_BASELINE_ITERS: int = 5000
    EVAL_BASELINE_SEEDS: int = 3
    EVAL_BASELINE_CACHE_MODE: str = "process"  # process | none
    EVAL_BASELINE_LABEL: str = "MCCFR-ES"
    EVAL_CFR_ITERATIONS: int = 5000  # compatibility alias; synced to EVAL_BASELINE_ITERS
    EVAL_ENABLE_RANDOM: bool = True
    EVAL_ENABLE_CFR: bool = False  # compatibility flag for solver baseline eval
    EVAL_ENABLE_NASH_CONV: bool = False
    BB_SIZE: float = 100.0

    # Checkpointing / logging
    CHECKPOINT_FREQ: int = 50
    MAX_CHECKPOINTS: int = 10
    CHECKPOINT_SINGLE_FILE: bool = False
    CHECKPOINT_SAVE_OPTIMIZER: bool = True
    CHECKPOINT_SAVE_SCHEDULER: bool = True
    CHECKPOINT_SAVE_LEAGUE: bool = True
    CHECKPOINT_SAVE_EXTRA: bool = True
    LOG_INTERVAL: int = 5
    RUN_DIR: str = "."
    RESUME_FROM: str = ""
    RUN_ID: str = ""

    WANDB_PROJECT: str = "alpha-holdem-poker"
    WANDB_RUN_NAME: str = ""
    WANDB_MODE: str = "offline"  # online | offline | disabled
    OFFLINE_LOGGING: bool = True

    # Misc
    SEED: int = 42
    DEVICE: str = field(default_factory=_default_device)

    def __post_init__(self):
        self.sync_legacy_fields()

    def sync_legacy_fields(self):
        """Keeps legacy field aliases consistent with the canonical config fields."""
        baseline_iters = int(getattr(self, "EVAL_BASELINE_ITERS", 5000))
        legacy_cfr_iters = int(getattr(self, "EVAL_CFR_ITERATIONS", baseline_iters))
        if legacy_cfr_iters != 5000 and baseline_iters == 5000:
            baseline_iters = legacy_cfr_iters
        self.EVAL_BASELINE_ITERS = int(baseline_iters)
        self.EVAL_CFR_ITERATIONS = int(baseline_iters)

        cache_mode = str(getattr(self, "EVAL_BASELINE_CACHE_MODE", "process")).lower()
        if cache_mode not in {"process", "none"}:
            cache_mode = "process"
        self.EVAL_BASELINE_CACHE_MODE = cache_mode

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not callable(v)
            and not isinstance(v, (classmethod, staticmethod))
        }

    def to_dict_instance(self) -> Dict[str, Any]:
        return dict(self.__dict__)
