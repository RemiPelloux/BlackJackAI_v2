import gym
from gym import spaces
import numpy as np
from game_logic import Game

class BlackjackEnv(gym.Env):
    def __init__(self):
        super(BlackjackEnv, self).__init__()
        self.game = Game(num_decks=6)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0: Stand, 1: Hit, 2: Double, 3: Bet

    def reset(self):
        self.game.start_round()
        state = self._get_state()
        return state

    def step(self, action):
        done = False
        reward = 0
        player_value = self.game.get_hand_value(self.game.player.hand)
        dealer_upcard_value = self.game.get_hand_value([self.game.dealer.hand[0]])

        if action == 0:  # Stand
            done = True
        elif action == 1:  # Hit
            self.game.player_action('hit')
            player_value = self.game.get_hand_value(self.game.player.hand)
            reward += self._hit_reward(player_value)
            if player_value > 21:  # Check for bust
                done = True
                reward -= 1.0  # Penalty for busting
            elif player_value >= 20:  # Penalty for hitting with 20 or more
                reward -= 0.5
        elif action == 2:  # Double
            if self.game.player.can_double_down():
                self.game.player.double_down()
                self.game.player_action('hit')
                player_value = self.game.get_hand_value(self.game.player.hand)
                done = True
                reward += self._double_reward(player_value)
                if player_value > 21:  # Check for bust after doubling
                    reward -= 2.0  # Heavier penalty for busting after doubling down
        elif action == 3:  # Bet
            bet_amount = self.game.player.money * np.random.uniform(0.01, 0.05)  # Random bet between 1% and 5%
            bet_amount = min(bet_amount, 500)  # Ensure max bet is 500
            self.game.player.place_bet(bet_amount)
            reward -= bet_amount * 0.01  # Small penalty for betting to encourage conservative betting

        if done:
            self.game.dealer_action()
            winner = self.game.check_winner()
            reward += self._calculate_reward(winner, player_value, dealer_upcard_value)
            if winner == 'player':
                reward += 10  # Cap the reward for winning
                self.game.player.money += 2 * self.game.player.current_bet  # Update balance with winnings
            elif winner == 'dealer':
                reward -= self.game.player.current_bet  # Deduct the bet amount as a penalty for losing
            elif winner == 'push':
                self.game.player.money += self.game.player.current_bet  # Return the bet amount for a push
            info = {
                'player_value': self.game.get_hand_value(self.game.player.hand),
                'dealer_value': self.game.get_hand_value(self.game.dealer.hand),
                'win_loss': winner,
                'bet_amount': self.game.player.current_bet
            }
            state = self._get_state()
        else:
            state = self._get_state()
            info = {'bet_amount': self.game.player.current_bet}

        return state, reward, done, info

    def _get_state(self):
        player_value = self.game.get_hand_value(self.game.player.hand)
        dealer_value = self.game.get_hand_value([self.game.dealer.hand[0]])
        usable_ace = int(any(card.value == 'A' for card in self.game.player.hand) and player_value + 10 <= 21)
        bankroll = self.game.player.money / 1000.0  # Normalize bankroll for state representation
        return np.array(
            [player_value / 32.0, dealer_value / 11.0, usable_ace, bankroll, self.game.player.current_bet / 500],
            dtype=np.float32)

    def _hit_reward(self, player_value):
        reward = 0
        if player_value <= 13:
            reward += 0.2  # Incentivize hitting with 13 or less
        elif player_value <= 16:
            reward += 0.1  # Lesser reward for hitting with <= 16
        elif player_value == 21:
            reward += 0.5  # Reward for hitting exactly 21
        elif player_value > 21:
            reward -= 1.0  # Penalty for busting
        if 11 < player_value <= 20:
            reward += 0.1  # Reward for having a strong hand after hitting
        if player_value >= 18:
            reward -= 10  # Penalty for hitting with 18 or more
        return reward

    def _double_reward(self, player_value):
        reward = 0
        if player_value > 21:
            reward -= 2.0  # Heavier penalty for busting after doubling down
        elif player_value == 21:
            reward += 1.0  # Reward for hitting exactly 21 after doubling down
        else:
            reward += 0.5  # Lesser reward for not busting after doubling down
        return reward

    def _calculate_reward(self, winner, player_value, dealer_upcard_value):
        reward = 0
        bet_amount = self.game.player.current_bet
        if winner == 'player':
            reward += 10  # Cap the reward for winning
            reward += min(bet_amount * 0.1, 20)  # Proportional reward based on bet, capped at 10
        elif winner == 'dealer':
            # Adjust penalty based on how close the player was to 21
            if player_value > 21:
                reward -= 1.0  # Penalty for busting
            elif 21 >= player_value >= 17:
                reward -= 0.5  # Lesser penalty for losing with a good hand
            else:
                reward -= 0.7  # Standard penalty for losing with a lower hand
        elif winner == 'push':
            reward += 0.5
        return reward

    def render(self, mode='human'):
        pass
