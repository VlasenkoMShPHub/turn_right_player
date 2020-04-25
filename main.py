from input import Env
from ddpg_algo import train_draft, get_tools
import pyautogui
import time
from q_learning import train_agent, QLearning, QLearningUpdater, EpsilonGreedyAgent


def main():
    env = Env()
    q_learning_agent = QLearning()
    updater = QLearningUpdater(q_learning_agent)
    epsilon_greedy = EpsilonGreedyAgent(q_learning_agent)
    rewards, steps = train_agent(env, epsilon_greedy, q_learning_agent, updater)


if __name__ == '__main__':
    time.sleep(1)
    main()
