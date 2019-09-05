from input import Env
from ddpg_algo import train_draft, get_tools
import pyautogui
import time


def main():
    env, noise, actor, critic, replay_buffer = get_tools()
    train_rewards, exploit_rewards, best_weights = train_draft(env, noise, actor, critic, replay_buffer)


if __name__ == '__main__':
    time.sleep(1)
    main()
