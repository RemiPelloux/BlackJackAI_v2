from blackjack_env import BlackjackEnv


def manual_test():
    env = BlackjackEnv()
    state = env.reset()
    done = False
    print(f"Initial state: {state}")

    while not done:
        print("Choose an action: 0 - Stand, 1 - Hit, 2 - Double")
        action = int(input())
        next_state, reward, done, info = env.step(action)
        print(f"Next state: {next_state}, Reward: {reward}, Done: {done}, Info: {info}")


if __name__ == "__main__":
    manual_test()
