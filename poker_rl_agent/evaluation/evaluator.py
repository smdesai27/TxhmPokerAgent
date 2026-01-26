from ..environment.openspiel_wrapper import PokerEnv
from ..evaluation.baseline_agents import RandomAgent
from ..environment.state_representation import StateEncoder
from ..algorithms.regret_matching import regret_matching_plus
import torch
import numpy as np

class Evaluator:
    def __init__(self, agent_model, device='cpu'):
        self.agent_model = agent_model
        self.device = device
        self.env = PokerEnv() # Config?
        self.encoder = StateEncoder(device)

    def evaluate(self, opponent, num_episodes=100):
        """
        Runs num_episodes games against opponent.
        Agent is Player 0 in half, Player 1 in half.
        Returns average return.
        """
        returns = []
        
        for i in range(num_episodes):
            self.env.reset()
            state = self.env.state
            
            # Agent plays as player 0 for first half
            agent_player_id = 0 if i < num_episodes / 2 else 1
            
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    action_list, probs = zip(*outcomes)
                    action = np.random.choice(action_list, p=probs)
                    state.apply_action(action)
                    continue
                
                current_player = state.current_player()
                
                if current_player == agent_player_id:
                    # Agent Move
                    with torch.no_grad():
                        state_tensor = self.encoder.encode_state(state, current_player, self.env.num_actions())
                        for k, v in state_tensor.items(): state_tensor[k] = v.unsqueeze(0)
                        
                        regrets = self.agent_model(state_tensor)
                        
                        # Mask illegal
                        legal_actions = state.legal_actions()
                        # Simple masking
                        all_regrets = regrets.cpu().numpy()[0]
                        legal_regrets = all_regrets[legal_actions]
                        # For evaluation, we can use RegretMatching or Greedy on Regret
                        # Using Greedy on positive regret is often better for exploitation
                        # But RM is standard.
                        
                        # Let's use argmax of regret (Exploit) or Average Policy?
                        # Since we only learned Regrets, we use RM strategy.
                        
                        # Manual Regret Matching for numpy
                        pos_regrets = np.maximum(legal_regrets, 0)
                        if pos_regrets.sum() > 0:
                            probs = pos_regrets / pos_regrets.sum()
                            action = np.random.choice(legal_actions, p=probs)
                        else:
                            action = np.random.choice(legal_actions)
                            
                    state.apply_action(action)
                else:
                    # Opponent Move
                    action = opponent.step(state)
                    state.apply_action(action)
            
            # Game Over
            r = state.returns()[agent_player_id]
            returns.append(r)
            
        return np.mean(returns)
