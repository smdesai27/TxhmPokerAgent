import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.distributions import Categorical
from poker_rl_agent.models.alpha_holdem_net import AlphaHoldemNetwork
from poker_rl_agent.environment.openspiel_wrapper import PokerEnv
from poker_rl_agent.environment.state_representation import StateEncoder
from poker_rl_agent.utils.config import Config
from poker_rl_agent.models.model_utils import masked_logits

def main():
    print("Welcome to AlphaHoldEm Implementation!")
    print("You are playing against the un-trained (or loaded) agent.")
    
    config = Config()
    env = PokerEnv(
        game_name=config.GAME_NAME,
        env_preset=config.ENV_PRESET,
        betting_abstraction=config.BETTING_ABSTRACTION,
    )
    num_actions = env.num_actions()
    
    model = AlphaHoldemNetwork(num_actions, config)
    # TODO: Load checkpoint if provided
    # model.load_state_dict...
    model.eval()
    
    encoder = StateEncoder(device="cpu", max_action_history=config.MAX_ACTION_HISTORY)
    
    env.reset()
    state = env.state
    
    while not state.is_terminal():
        print("\n" + "="*30)
        print(f"Current Player: {state.current_player()}")
        print(f"State: {str(state)}") # Simplified display
        
        if state.is_chance_node():
            print("Dealing cards...")
            outcomes = state.chance_outcomes()
            action_list, probs = zip(*outcomes)
            action = np.random.choice(action_list, p=probs)
            state.apply_action(action)
            continue
            
        legal_actions = state.legal_actions()
        
        if state.current_player() == 0:
            # Human Player (Let's say Human is P0)
            print(f"Your Turn! Legal Actions: {legal_actions}")
            # Try to interpret actions if possible, else ask for index
            try:
                choice = int(input(f"Enter action index {legal_actions}: "))
                if choice not in legal_actions:
                    print("Invalid action. Picking random.")
                    choice = np.random.choice(legal_actions)
            except:
                 print("Invalid input. Picking random.")
                 choice = np.random.choice(legal_actions)
            state.apply_action(choice)
        else:
            # Agent Player (P1)
            print("Agent is thinking...")
            with torch.no_grad():
                state_tensor = encoder.encode_state(state, 1, num_actions)
                for k, v in state_tensor.items():
                    state_tensor[k] = v.unsqueeze(0)
                
                outputs = model(state_tensor)
                logits = masked_logits(outputs["policy_logits"], state_tensor["legal_action_mask"])
                action = int(Categorical(logits=logits).sample().item())
                    
            print(f"Agent chose action: {action}")
            state.apply_action(action)
            
    print("\nGame Over!")
    print(f"Returns: {state.returns()}")

if __name__ == "__main__":
    main()
