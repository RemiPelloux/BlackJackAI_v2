import random
import json

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def __str__(self):
        return f"{self.value} of {self.suit}"

    def get_value(self):
        if self.value in ['J', 'Q', 'K']:
            return 10
        if self.value == 'A':
            return 11
        return int(self.value)

class Deck:
    def __init__(self, num_decks=1):
        self.num_decks = num_decks
        self.cards = []
        self.build_deck()

    def build_deck(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [Card(suit, value) for suit in suits for value in values] * self.num_decks
        random.shuffle(self.cards)

    def draw_card(self):
        if len(self.cards) == 0:
            self.build_deck()
        return self.cards.pop()

class Player:
    def __init__(self, name, money):
        self.name = name
        self.money = money
        self.hand = []
        self.split_hand = []
        self.current_bet = 0
        self.split_bet = 0
        self.doubled_down = False

    def receive_card(self, card, to_split_hand=False):
        if to_split_hand:
            self.split_hand.append(card)
        else:
            self.hand.append(card)

    def clear_hand(self):
        self.hand = []
        self.split_hand = []
        self.doubled_down = False

    def place_bet(self, amount):
        self.current_bet = amount
        self.money -= amount

    def place_split_bet(self):
        self.split_bet = self.current_bet
        self.money -= self.split_bet

    def can_split(self):
        return len(self.hand) == 2 and self.hand[0].value == self.hand[1].value

    def can_double_down(self):
        total = sum(card.get_value() for card in self.hand)
        return total in [9, 10, 11]

    def double_down(self):
        if self.can_double_down():
            self.money -= self.current_bet
            self.current_bet *= 2
            self.doubled_down = True
        else:
            print("Cannot double down on this hand.")

    def adjust_money(self, amount):
        self.money += amount

class Game:
    def __init__(self, num_decks=1):
        self.deck = Deck(num_decks)
        self.player = Player("Player", self.load_player_money())
        self.dealer = Player("Dealer", 0)

    def load_player_money(self):
        try:
            with open('player_data.json', 'r') as file:
                data = json.load(file)
            return data.get('player_money', 10000)
        except FileNotFoundError:
            return 10000

    def save_player_money(self):
        with open('player_data.json', 'w') as file:
            data = {'player_money': self.player.money}
            json.dump(data, file)

    def start_round(self):
        self.player.clear_hand()
        self.dealer.clear_hand()
        self.player.receive_card(self.deck.draw_card())
        self.player.receive_card(self.deck.draw_card())
        self.dealer.receive_card(self.deck.draw_card())
        self.dealer.receive_card(self.deck.draw_card())

    def player_action(self, action):
        if action == 'hit':
            self.player.receive_card(self.deck.draw_card())
        elif action == 'stand':
            return False
        elif action == 'split':
            if not self.player.can_split():
                print("You can't split this hand.")
                return True
            self.player.place_split_bet()
            self.player.split_hand.append(self.player.hand.pop())
            self.player.receive_card(self.deck.draw_card())
            self.player.receive_card(self.deck.draw_card(), to_split_hand=True)
        elif action == 'double':
            if not self.player.can_double_down():
                print("You can't double down on this hand.")
                return True
            self.player.double_down()
            self.player.receive_card(self.deck.draw_card())
            return False
        return True

    def dealer_action(self):
        while self.get_hand_value(self.dealer.hand) < 17:
            self.dealer.receive_card(self.deck.draw_card())

    def get_hand_value(self, hand):
        value = sum(card.get_value() for card in hand)
        aces = sum(1 for card in hand if card.value == 'A')
        while value > 21 and aces:
            value -= 10
            aces -= 1
        return value

    def check_winner(self):
        player_value = self.get_hand_value(self.player.hand)
        dealer_value = self.get_hand_value(self.dealer.hand)
        if player_value > 21:
            return 'dealer'
        if dealer_value > 21 or player_value > dealer_value:
            return 'player'
        if player_value == dealer_value:
            return 'push'
        return 'dealer'

    def resolve_bets(self, winner):
        player = self.player
        if winner == 'player':
            if self.get_hand_value(player.hand) == 21 and len(player.hand) == 2:
                player.adjust_money(2.5 * player.current_bet)  # Blackjack pays 3:2
            else:
                player.adjust_money(2 * player.current_bet)  # Standard win
        elif winner == 'push':
            player.adjust_money(player.current_bet)  # Push returns the bet
        # No adjustment for loss; the bet is already subtracted
        self.save_player_money()
