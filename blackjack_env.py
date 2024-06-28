import gym
from gym import spaces
import numpy as np
from game_logic import Game

class BlackjackEnv(gym.Env):
    def __init__(self):
        super(BlackjackEnv, self).__init__()
        self.game = Game(num_decks=6)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: Stand, 1: Hit, 2: Double
        self.min_bet = 10  # Minimum bet amount
        self.max_bet = 500  # Maximum bet amount

    def reset(self):
        # Place bet before starting the game using a progressive betting strategy
        bet_amount = self.progressive_betting_strategy()
        self.game.player.place_bet(bet_amount)

        self.game.start_round()
        state, done, reward = self._check_immediate_blackjack()
        if done:
            return state  # Immediate return if round is done due to Blackjack

        state = self._get_state()
        return state

    def step(self, action):
        done = False
        reward = 0
        player_value = self.game.get_hand_value(self.game.player.hand)
        dealer_upcard_value = self.game.get_hand_value([self.game.dealer.hand[0]])
        has_ace = any(card.value == 'A' for card in self.game.player.hand)

        if action == 0:  # Stand
            if player_value < 11:
                reward -= 0.5  # Small penalty for not hitting if hand value is less than 11
            if has_ace and player_value < 15:
                reward -= 0.5  # Small penalty for not hitting if hand value is less than 15 and has an Ace
            done = True
        elif action == 1:  # Hit
            self.game.player_action('hit')
            player_value = self.game.get_hand_value(self.game.player.hand)
            reward += self._hit_reward(player_value)
            if has_ace and player_value < 15:
                reward += 0.5  # Reward for hitting if hand value is less than 15 and has an Ace
            if player_value > 21:  # Check for bust
                done = True
                reward -= 1.0  # Penalty for busting
        elif action == 2:  # Double
            if self.game.player.can_double_down():
                self.game.player.double_down()
                self.game.player_action('hit')
                player_value = self.game.get_hand_value(self.game.player.hand)
                done = True
                reward += self._double_reward(player_value)
                if player_value > 21:  # Check for bust after doubling
                    reward -= 2.0  # Penalty for busting after doubling down

        if done:
            dealer_value = self.game.get_hand_value(self.game.dealer.hand)
            if dealer_value <= player_value:
                self.game.dealer_action()
            winner = self.game.check_winner()
            return self._end_round(winner)

        state = self._get_state()
        info = {'bet_amount': self.game.player.current_bet}
        return state, reward, done, info

    def progressive_betting_strategy(self):
        balance = self.game.player.money
        base_bet = self.min_bet
        if balance > 10000:
            bet_amount = base_bet * 2  # Double the base bet if balance is high
        elif balance > 5000:
            bet_amount = base_bet * 1.5  # 1.5 times the base bet if balance is medium
        else:
            bet_amount = base_bet  # Base bet if balance is low
        bet_amount = min(max(bet_amount, self.min_bet), self.max_bet)  # Ensure bet is between min_bet and max_bet
        return np.floor(bet_amount)  # Floor the bet amount to ensure it is a whole number

    def _check_immediate_blackjack(self):
        player_value = self.game.get_hand_value(self.game.player.hand)
        dealer_upcard_value = self.game.get_hand_value([self.game.dealer.hand[0]])
        done = False
        reward = 0

        if player_value == 21:
            dealer_value = self.game.get_hand_value(self.game.dealer.hand)
            if dealer_value == 21:
                winner = 'push'
            else:
                winner = 'player'
            state, reward, done, info = self._end_round(winner)
            return state, done, reward

        return self._get_state(), done, reward

    def _end_round(self, winner):
        player_value = self.game.get_hand_value(self.game.player.hand)
        dealer_value = self.game.get_hand_value(self.game.dealer.hand)
        reward = self._calculate_reward(winner, player_value, dealer_value)
        if winner == 'player':
            self.game.player.money += 2 * self.game.player.current_bet  # Update balance with winnings
        elif winner == 'dealer':
            max_penalty = min(self.game.player.current_bet * 0.05, 5)
            reward -= max_penalty  # Proportional penalty for losing
        elif winner == 'push':
            self.game.player.money += self.game.player.current_bet  # Return the bet amount for a push
        state = self._get_state()
        info = {
            'player_value': player_value,
            'dealer_value': dealer_value,
            'win_loss': winner,
            'bet_amount': self.game.player.current_bet
        }
        done = True
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
            reward += 1.0  # Reduced reward for hitting with 13 or less
        elif player_value <= 16:
            reward += 0.3  # Lesser reward for hitting with <= 16
        elif player_value == 21:
            reward += 2.0  # Reduced reward for hitting exactly 21
        elif player_value > 21:
            reward -= 1.0  # Reduced penalty for busting
        if 11 < player_value <= 20:
            reward += 0.2  # Lesser reward for having a strong hand after hitting
        if player_value >= 18:
            reward -= 1  # Reduced penalty for hitting with 18 or more
        return reward

    def _double_reward(self, player_value):
        reward = 0
        if player_value > 21:
            reward -= 3.0  # Reduced penalty for busting after doubling down
        elif player_value == 21:
            reward += 1.0  # Reduced reward for hitting exactly 21 after doubling down
        else:
            reward += 0.2  # Lesser reward for not busting after doubling down
        return reward

    def _calculate_reward(self, winner, player_value, dealer_upcard_value):
        reward = 0
        bet_amount = self.game.player.current_bet
        if winner == 'player':
            reward += 5  # Reduced cap for the reward for winning
            reward += min(bet_amount * 0.05, 10)  # Reduced proportional reward based on bet, capped at 10
        elif winner == 'dealer':
            # Adjust penalty based on how close the player was to 21
            max_penalty = min(self.game.player.money * 0.05, 5)  # Cap penalty to 5% of wallet or 5
            if player_value > 21:
                reward -= 2.0  # Reduced penalty for busting
            elif 21 >= player_value >= 17:
                reward -= 0.5  # Lesser penalty for losing with a good hand
            elif 11 > player_value >= 3:
                reward -= max_penalty  # Reduced penalty for losing with a very low hand
            else:
                reward -= 2.0  # Standard penalty for losing with a lower hand
        elif winner == 'push':
            reward += 0.5  # Reward for a push
        return reward

    def render(self, mode='human'):
        pass

# Other required changes to the game logic or main training loop might be needed to fully support these updates.
