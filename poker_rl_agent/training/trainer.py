import torch
import torch.optim as optim
import os
import copy
import wandb
from ..models.alpha_holdem_net import AlphaHoldemNetwork
from ..algorithms.self_play import SelfPlayWorker
from ..training.replay_buffer import ReplayBuffer
from ..training.checkpointing import save_checkpoint, load_checkpoint
from ..environment.openspiel_wrapper import PokerEnv
from ..utils.config import Config
from ..utils.logging_utils import log_metrics, init_wandb

class Trainer:
    def __init__(self, config=None):
        self.config = config if config else Config()
        self.device = self.config.DEVICE
        
        # Env
        self.env = PokerEnv(self.config.GAME_NAME)
        num_actions = self.env.num_actions()
        
        # Models
        print(f"Initializing Model on {self.device}")
        self.model = AlphaHoldemNetwork(num_actions, self.config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LR)
        
        # Target Network
        if self.config.USE_TARGET_NET:
            self.target_model = copy.deepcopy(self.model)
            self.target_model.to(self.device)
            self.target_model.eval()
        else:
            self.target_model = self.model
            
        # Scheduler
        if self.config.LR_SCHEDULER == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=1000
            )
        else:
            self.scheduler = None
        
        # Buffer
        self.buffer = ReplayBuffer(self.config.BUFFER_SIZE)
        
        # Workers - Use TARGET model for data collection to stabilize policy
        self.worker = SelfPlayWorker(self.env, self.target_model, self.device)
        
    def _soft_update_target(self):
        """Soft update target network parameters."""
        tau = self.config.TARGET_UPDATE_TAU
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train(self):
        init_wandb(config=self.config)
        step = 0
        
        # Load Checkpoint if exists
        if os.path.exists("checkpoints"):
             # Find latest
            pass # TODO: Implement load logic if needed
        
        print(f"Starting Training with {self.config.CFR_ITERATIONS} iterations...")

        for iteration in range(self.config.CFR_ITERATIONS):
            # 1. Self Play / Data Collection
            for _ in range(self.config.SELF_PLAY_GAMES):
                for player in range(self.config.PLAYERS):
                    samples = self.worker.run_external_sampling(player)
                    for s_tensor, regrets in samples:
                        self.buffer.add(s_tensor, None, regrets, None)

            # 2. Train Network
            if len(self.buffer) < self.config.BATCH_SIZE:
                continue
                
            loss_sum = 0
            grad_norm_sum = 0
            pad_config = {'action_history': self.env.num_actions()}
            
            # Train steps per iteration (can be config driven)
            TRAIN_STEPS = 10 
            
            for _ in range(TRAIN_STEPS):
                state_dicts, _, target_regrets, _ = self.buffer.sample(self.config.BATCH_SIZE, pad_config)
                
                # Move to device
                for k, v in state_dicts.items():
                    if k != 'action_history':
                         if v.dim() == 3 and v.size(1) == 1:
                            v = v.squeeze(1)
                    state_dicts[k] = v.to(self.device)
                
                target_regrets = target_regrets.to(self.device)
                
                # Norm Target Regrets (Phase 3 Fix)
                # (regrets - mean) / (std + 1e-8)
                regret_mean = target_regrets.mean()
                regret_std = target_regrets.std() + 1e-8
                target_regrets = (target_regrets - regret_mean) / regret_std
                
                self.optimizer.zero_grad()
                pred_regrets = self.model(state_dicts)
                
                loss = torch.nn.functional.mse_loss(pred_regrets, target_regrets)
                
                # SAFETY: Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"CRITICAL: Loss is NaN/Inf at step {step}!")
                    # Check inputs/targets
                    print(f"Target Regrets stats: Min={target_regrets.min()}, Max={target_regrets.max()}, Mean={target_regrets.mean()}")
                    # Zero out loss to prevent crash? Or raise?
                    # For emergency fix, we raise to stop bad training
                    raise ValueError("Loss is NaN/Inf")

                loss.backward()
                
                # Gradient Clipping (Phase 3 Fix)
                # Calculate unclipped norm for logging
                total_norm_unclipped = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm_unclipped += p.grad.data.norm(2).item() ** 2
                total_norm_unclipped = total_norm_unclipped ** 0.5

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                grad_norm_sum += grad_norm.item()
                
                # SAFETY: Check for NaN/Inf gradients
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                     raise ValueError(f"Gradient is NaN/Inf: {grad_norm}")
                
                self.optimizer.step()
                
                if self.config.USE_TARGET_NET:
                    self._soft_update_target()
                    
                loss_sum += loss.item()
            
            # Valid / Scheduler step
            avg_loss = loss_sum / TRAIN_STEPS
            if self.scheduler:
                self.scheduler.step(avg_loss)

            step += 1
            if step % self.config.LOG_INTERVAL == 0:
                metrics = {
                    'loss': avg_loss,
                    'grad_norm_clipped': grad_norm_sum / TRAIN_STEPS,
                    # We only captured unclipped for the last step in the loop above really, 
                    # but good enough for diagnosis if we move it out or average it.
                    # For now just log clipped.
                    'iteration': iteration,
                    'buffer_size': len(self.buffer),
                    'lr': self.optimizer.param_groups[0]['lr'],
                    # Input/target stats (from last batch)
                    'target_regret_mean': regret_mean.item(),
                    'target_regret_std': regret_std.item()
                }
                
                # Log weight norms periodically
                if step % 100 == 0:
                     for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            metrics[f'weight_norm/{name}'] = param.norm().item()
                            
                log_metrics(metrics, step)
            
            if step % self.config.CHECKPOINT_FREQ == 0:
                path = f"checkpoints/point_{step}.pt"
                save_checkpoint(self.model, self.optimizer, step, None, path, max_keep=self.config.MAX_CHECKPOINTS)
                print(f"Saved checkpoint at step {step}")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
