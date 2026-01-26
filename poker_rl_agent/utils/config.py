import torch

class Config:
    # Game Settings
    GAME_NAME = "universal_poker"
    PLAYERS = 2
    
    # Neural Network Architecture
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 256
    NUM_LAYERS_CARD = 3
    NUM_LAYERS_ACTION = 2
    DROPOUT = 0.1
    
    # Training Hyperparameters
    LR = 1e-4
    BATCH_SIZE = 128
    CFR_ITERATIONS = 1000
    SELF_PLAY_GAMES = 4  # K games in parallel
    BUFFER_SIZE = 100000
    
    # Checkpointing & Logging
    CHECKPOINT_FREQ = 100
    EVAL_FREQ = 50
    LOG_INTERVAL = 10
    SEED = 42
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith("__") 
                and not isinstance(v, (classmethod, staticmethod)) 
                and not callable(v)
                and k.isupper()} # Only export uppercase config vars

