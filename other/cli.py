# cli.py
import os


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def display_message(message):
    print("\n" + "=" * len(message))
    print(message)
    print("=" * len(message))


def get_player_action(can_split=False, can_double=False):
    actions = ['hit', 'stand']
    if can_split:
        actions.append('split')
    if can_double:
        actions.append('double')
    action = ''
    while action not in actions:
        print("\nAvailable actions: ")
        for act in actions:
            print(f" - {act.capitalize()}")
        action = input("\nChoose an action: ").lower()
    return action


def get_bet_amount(player_money):
    while True:
        try:
            bet = int(input(f"\nEnter your bet amount (Available money: ${player_money}): $"))
            if 2 <= bet <= min(500, player_money):
                return bet
            else:
                print("Invalid bet amount. The bet must be between $2 and $500, and you must have enough money.")
        except ValueError:
            print("Please enter a valid number.")


def ask_play_again():
    return input("\nPlay another round? (y/n): ").lower() == 'y'


def display_hands(player, dealer, initial=False):
    clear_screen()
    print("\nDealer's Hand:")
    if initial:
        print(f" [ {dealer.hand[0]} , Hidden ]")
    else:
        print(" [" + " , ".join(str(card) for card in dealer.hand) + " ]")

    print("\nPlayer's Hand:")
    print(" [" + " , ".join(str(card) for card in player.hand) + " ]")
    print(f"\nPlayer's money: ${player.money}")
    if player.split_hand:
        print("\nPlayer's Split Hand:")
        print(" [" + " , ".join(str(card) for card in player.split_hand) + " ]")


def display_result(winner):
    result_message = f"\n{winner.capitalize()} wins!"
    display_message(result_message)
