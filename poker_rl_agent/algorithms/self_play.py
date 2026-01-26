import torch
import copy
from ..environment.openspiel_wrapper import PokerEnv
from ..environment.state_representation import StateEncoder
from .regret_matching import regret_matching_plus, get_action_from_strategy

class SelfPlayWorker:
    def __init__(self, env: PokerEnv, model, device='cpu'):
        self.env = env
        self.model = model
        self.device = device
        self.encoder = StateEncoder(device)

    def generate_trajectory(self):
        """
        Plays one game (episode) using current model.
        Returns list of (State, Action, Reward, LegalActions, etc.) for traversal.
        
        Note: DeepCFR usually needs traversal of the game tree (External Sampling)
        rather than just a single self-play path to compute counterfactuals effectively.
        However, if we just run self-play, we might be doing Reinforcement Learning (DQN/PPO styling)
        or we need to branch at each node to compute regrets.
        
        For correct CFR+ with NN, we need Outcome Sampling or External Sampling.
        Simple self-play trajectory is insufficient for Counterfactual Regret computation
        unless we have a Value Network to estimate values of unchosen actions.
        
        Given AlphaHoldEm Architecture, it likely uses an Advantage/Regret Network.
        We will implement External Sampling (MCCFR) here.
        This effectively means 'playing' against the opponent but traversing all my own actions
        to compute regrets.
        """
        pass

    def external_sampling_traversal(self, state, model_net, traversing_player, samples_accumulator=None):
        """
        Recursive function for MCCFR External Sampling.
        Returns: Utility (value) of the state.
        """
        if state.is_terminal():
            # Return payoff for traversing_player
            return state.returns()[traversing_player]
        
        if state.is_chance_node():
            # Sample chance outcome
            # OpenSpiel chance handling: apply random action
            outcomes = state.chance_outcomes()
            action_list, probs = zip(*outcomes)
            action = np.random.choice(action_list, p=probs)
            state.apply_action(action)
            return self.external_sampling_traversal(state, model_net, traversing_player, samples_accumulator)
            
        current_player = state.current_player()
        legal_actions = state.legal_actions()
        
        # Get Strategy from Network
        # Encode state
        num_actions = self.env.num_actions()
        state_tensor = self.encoder.encode_state(state, current_player, num_actions)
        # Add batch dim
        for k, v in state_tensor.items():
            state_tensor[k] = v.unsqueeze(0)
            
        with torch.no_grad():
            regret_pred = model_net(state_tensor) # [1, NumActions]
            
        # Mask illegal actions
        all_actions_mask = torch.ones_like(regret_pred) * -1e9
        all_actions_mask[0, legal_actions] = 0
        regret_pred = regret_pred + all_actions_mask
        
        strategy = regret_matching_plus(regret_pred).cpu().numpy()[0]
        # Normalize over legal actions just in case
        legal_strategy = strategy[legal_actions]
        if legal_strategy.sum() == 0:
            legal_strategy = np.ones_like(legal_strategy) / len(legal_strategy)
        else:
            legal_strategy /= legal_strategy.sum()
            
        if current_player == traversing_player:
            # Traverse ALL actions (External Sampling)
            # We want to compute regrets for this state
            values = {}
            for idx, action in enumerate(legal_actions):
                next_state = state.clone()
                next_state.apply_action(action)
                # Recurse
                values[action] = self.external_sampling_traversal(next_state, model_net, traversing_player, samples_accumulator)
            
            # Compute Utility of this node using the current strategy
            node_utility = sum(legal_strategy[i] * values[act] for i, act in enumerate(legal_actions))
            
            # Compute Regrets
            # Regret = Q(s, a) - V(s)
            state_regrets = torch.zeros(self.env.num_actions())
            for idx, action in enumerate(legal_actions):
                r = values[action] - node_utility
                state_regrets[action] = r
                
            # Store (state_tensor, state_regrets)
            # We assume samples_accumulator is a list passed in
            if samples_accumulator is not None:
                samples_accumulator.append((state_tensor, state_regrets))
            
            return node_utility
            
        else:
            # Opponent: Sample ONE action based on strategy
            action_idx = np.random.choice(len(legal_actions), p=legal_strategy)
            action = legal_actions[action_idx]
            
            next_state = state.clone()
            next_state.apply_action(action)
            return self.external_sampling_traversal(next_state, model_net, traversing_player, samples_accumulator)

    def run_external_sampling(self, traversing_player):
        """Runs one episode of external sampling for the traversing player."""
        self.env.reset()
        samples = []
        self.external_sampling_traversal(self.env.state, self.model, traversing_player, samples)
        return samples

import numpy as np
