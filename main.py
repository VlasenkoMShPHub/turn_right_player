from input import Env
import time
from q_learning import train_agent, QLearning, QLearningUpdater, EpsilonGreedyAgent, play_agent
import numpy as np


load_table_name = 'q_l_tables/Q_learning_265_54031_54056.npy'
learn = True


def main():
    env = Env()
    q_learning_agent = QLearning()
    if len(load_table_name) > 20:
        print(f'loading table {load_table_name}')
        q_learning_agent.load_table(load_table_name)
    print(env.check_car_pos())
    assert env.check_car_pos() == (269, 116)
    updater = QLearningUpdater(q_learning_agent)
    epsilon_greedy = EpsilonGreedyAgent(q_learning_agent)
    if learn:
        rewards = train_agent(env, epsilon_greedy, q_learning_agent, updater)
        np.save('q_l_tables/rewards.npy', rewards)
    else:
        print('exploiting')
        play_agent(env, q_learning_agent)


if __name__ == '__main__':
    time.sleep(1)
    main()
