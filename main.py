from input import Env
from ddpg_algo import train_draft, get_tools
import pyautogui
import time
from q_learning import train_agent, QLearning, QLearningUpdater, EpsilonGreedyAgent

load_table_name = 'q_l_tables/.npy'


def main():
    env = Env()
    q_learning_agent = QLearning()
    if len(load_table_name) > 20:
        print(f'loading table {load_table_name}')
        q_learning_agent.load_table(load_table_name)
    updater = QLearningUpdater(q_learning_agent)
    epsilon_greedy = EpsilonGreedyAgent(q_learning_agent)
    rewards = train_agent(env, epsilon_greedy, q_learning_agent, updater)


if __name__ == '__main__':
    main()
