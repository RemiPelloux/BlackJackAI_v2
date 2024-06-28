import torch
import numpy as np
import json
from blackjack_env import BlackjackEnv
from dqn_blackjack import Agent


def log_agent_behavior(agent, env, num_games=20, log_file='agent_behavior_log.json'):
    game_data = []
    for game in range(num_games):
        state = env.reset()
        state = np.array(state).reshape(1, -1)  # Ensure state is a 2D array
        game_log = {'game': game, 'steps': []}
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            dealer_cards = [str(card) for card in env.game.dealer.hand]
            player_cards = [str(card) for card in env.game.player.hand]
            game_log['steps'].append({
                'state': state.tolist(),
                'action': action,
                'reward': reward,
                'next_state': next_state.tolist(),
                'done': done,
                'dealer_cards': dealer_cards,
                'player_cards': player_cards,
                'bet': env.game.player.current_bet,
                'win_loss': 'win' if reward > 0 else 'loss' if reward < 0 else 'tie'
            })
            state = np.array(next_state).reshape(1, -1)
            if done:
                break
        game_data.append(game_log)

    with open(log_file, 'w') as f:
        json.dump(game_data, f, indent=4)


if __name__ == "__main__":
    env = BlackjackEnv()
    state_size = env.observation_space.shape[0]
    action_size = 13  # 10 bet sizes + 3 moves
    agent = Agent(state_size, action_size)
    agent.load("dqn_blackjack.pth")  # Load the trained model

    # Log the agent's behavior for 20 games
    log_agent_behavior(agent, env, num_games=20, log_file='agent_behavior_log.json')
