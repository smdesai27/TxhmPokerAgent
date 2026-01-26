import torch
import torch.optim as optim
import os
from ..models.alpha_holdem_net import AlphaHoldemNetwork
from ..algorithms.self_play import SelfPlayWorker
from ..training.replay_buffer import ReplayBuffer
from ..training.checkpointing import save_checkpoint, load_checkpoint
from ..environment.openspiel_wrapper import PokerEnv
from ..utils.config import Config
from ..utils.logging_utils import log_metrics, init_wandb

class Trainer:
    def __init__(self, config=None):
        self.config = config if config else Config
        self.device = self.config.DEVICE
        
        # Env
        self.env = PokerEnv(self.config.GAME_NAME)
        num_actions = self.env.num_actions()
        
        # Model
        self.model = AlphaHoldemNetwork(num_actions, self.config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LR)
        
        # Buffer
        self.buffer = ReplayBuffer(self.config.BUFFER_SIZE)
        
        # Workers (For now simple sequential self play)
        self.worker = SelfPlayWorker(self.env, self.model, self.device)
        
    def train(self):
        init_wandb()
        step = 0
        
        # Load Checkpoint if exists
        # step = load_checkpoint(...)

        for iteration in range(self.config.CFR_ITERATIONS):
            # 1. Self Play / Data Collection
            for _ in range(self.config.SELF_PLAY_GAMES):
                # Iterate over players to traverse
                for player in range(self.config.PLAYERS):
                    samples = self.worker.run_external_sampling(player)
                    for s_tensor, regrets in samples:
                        # Convert dict to cpu for buffer storage to save GPU memory if needed
                        # But here we keep as is
                        self.buffer.add(s_tensor, None, regrets, None) # action/reward unused for regret regression

            # 2. Train Network
            # Sample batch
            if len(self.buffer) < self.config.BATCH_SIZE:
                continue
                
            loss_sum = 0
            pad_config = {'action_history': self.env.num_actions()}
            
            for _ in range(10): # Train steps per iteration
                state_dicts, _, target_regrets, _ = self.buffer.sample(self.config.BATCH_SIZE, pad_config)
                
                # Move to device
                for k, v in state_dicts.items():
                    # If it was padded (action_history), it is already [B, MaxLen]
                    # If it was stacked (others), it was [B, 1, Dim] -> Squeeze
                    if k != 'action_history':
                         if v.dim() == 3 and v.size(1) == 1:
                            v = v.squeeze(1)
                    
                    state_dicts[k] = v.to(self.device)
                
                target_regrets = target_regrets.to(self.device)
                
                self.optimizer.zero_grad()
                pred_regrets = self.model(state_dicts)
                
                # Loss: MSE between predicted cumulative/average regret and sampled instantaneous regret
                # Note: DeepCFR typically adds sampled regret to a memory and trains on the SUM/Average.
                # Here we train on the instantaneous regret sample (high variance but unbiased mean).
                # Better: Use reservoir sampling or aggregate.
                loss = torch.nn.functional.mse_loss(pred_regrets, target_regrets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                loss_sum += loss.item()

            step += 1
            if step % self.config.LOG_INTERVAL == 0:
                log_metrics({'loss': loss_sum / 10, 'iteration': iteration}, step)
            
            if step % self.config.CHECKPOINT_FREQ == 0:
                if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
                save_checkpoint(self.model, self.optimizer, step, None, f"checkpoints/point_{step}.pt")
                print(f"Saved checkpoint at step {step}")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
