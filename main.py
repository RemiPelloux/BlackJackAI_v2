from dqn_agent import DQNAgent
from blackjack_env import BlackjackEnv
from utils import plot_stats

def train_dqn(episodes, model_path=None):
    env = BlackjackEnv()
    agent = DQNAgent(env)
    batch_size = 64
    rewards = []
    cumulative_rewards = []
    total_reward = 0
    balances = []
    total_wins = 0
    total_losses = 0
    player_values = []
    dealer_values = []
    actions_taken = []
    bet_amounts = []

    if model_path:
        agent.load(model_path)

    for e in range(episodes):
        state = env.reset()
        episode_reward = 0
        for time in range(500):  # Limit each episode to 500 steps
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            player_values.append(info.get('player_value', 0))
            dealer_values.append(info.get('dealer_value', 0))
            actions_taken.append(action)
            bet_amounts.append(info.get('bet_amount', 0))
            if done:
                break
        total_reward += episode_reward
        rewards.append(episode_reward / (time + 1))
        cumulative_rewards.append(total_reward)
        balances.append(env.game.player.money)

        if 'win_loss' in info:
            if info['win_loss'] == 'player':
                total_wins += 1
            elif info['win_loss'] == 'dealer':
                total_losses += 1

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if e % 100 == 0 and e > 0:
            print('--------------------------------------------------')
            print(f"Episode: {e}, Reward: {rewards[-1]:.2f}, Balance: {balances[-1]:.2f}, Wins: {total_wins},"
                  f" Losses: {total_losses}, winrate: {total_wins / (total_wins + total_losses):.2f}, bet_amount: {bet_amounts[-1]},"
                  f" , player_value: {player_values[-1]}, dealer_value: {dealer_values[-1]}")
            print(f"Player Cards: {[card.__str__() for card in env.game.player.hand]}")
            print(f"Dealer Upcard:{[card.__str__() for card in env.game.dealer.hand]}")
            print('--------------------------------------------------')

    agent.save("final_model.pth")
    plot_stats(rewards, cumulative_rewards, balances, total_wins, total_losses, player_values, dealer_values, actions_taken, bet_amounts, episodes, env)


if __name__ == "__main__":
    episodes = 1000
    model_path = "final_model.pth"
    train_dqn(episodes, model_path)
