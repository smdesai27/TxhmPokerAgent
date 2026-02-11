import torch
from dataclasses import dataclass, field

@dataclass
class Config:
    # Game Settings
    GAME_NAME: str = "universal_poker"
    PLAYERS: int = 2
    
    # Neural Network Architecture
    EMBEDDING_DIM: int = 64
    HIDDEN_DIM: int = 128 # Reduced from 512
    NUM_LAYERS_CARD: int = 4
    NUM_LAYERS_ACTION: int = 2
    DROPOUT: float = 0.1
    USE_LAYERNORM: bool = True
    USE_RESIDUAL: bool = True 
    INIT_GAIN: float = 0.1
    
    # Training Hyperparameters
    LR: float = 5e-5 # Reduced from 5e-4
    LR_SCHEDULER: str = "constant" # plateau, cosine, or none
    BATCH_SIZE: int = 64 # Reduced from 512
    CFR_ITERATIONS: int = 100000
    SELF_PLAY_GAMES: int = 20
    BUFFER_SIZE: int = 500000
    
    # Optimization
    OPTIMIZER: str = "adam"
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 0.5 # New conservative clip
    REGRET_CLIP: float = 10000.0 # New regret clip range
    USE_TARGET_NET: bool = True
    TARGET_UPDATE_TAU: float = 0.005
    TARGET_UPDATE_FREQ: int = 1
    
    # Checkpointing & Logging
    CHECKPOINT_FREQ: int = 500
    MAX_CHECKPOINTS: int = 5
    EVAL_FREQ: int = 1000
    LOG_INTERVAL: int = 10
    SEED: int = 42
    
    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    @classmethod
    def to_dict(cls):
        # Helper to convert dataclass or class to dict
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith("__") 
                and not callable(v) 
                and not isinstance(v, (classmethod, staticmethod))}

    def to_dict_instance(self):
        return self.__dict__
