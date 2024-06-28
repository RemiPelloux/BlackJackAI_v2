import matplotlib.pyplot as plt


def plot_stats(rewards, cumulative_rewards, balances, total_wins, total_losses, player_values, dealer_values, actions_taken, bet_amounts, episodes, env):
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')

    plt.subplot(2, 3, 2)
    plt.plot(cumulative_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward')

    plt.subplot(2, 3, 3)
    plt.plot(balances)
    plt.xlabel('Episodes')
    plt.ylabel('Balance')
    plt.title('Player Balance Over Time')

    plt.subplot(2, 3, 4)
    plt.bar(['Wins', 'Losses'], [total_wins, total_losses])
    plt.title('Total Wins vs Losses')

    plt.subplot(2, 3, 5)
    plt.plot(bet_amounts)
    plt.xlabel('Episodes')
    plt.ylabel('Bet Amount')
    plt.title('Bet Amounts Over Time')

    plt.subplot(2, 3, 6)
    plt.plot(player_values, label='Player Hand Value')
    plt.plot(dealer_values, label='Dealer Hand Value')
    plt.xlabel('Episodes')
    plt.ylabel('Hand Value')
    plt.legend()
    plt.title('Player and Dealer Hand Values Over Episodes')

    plt.tight_layout()
    plt.show()
