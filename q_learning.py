import time
import copy
import numpy as np
import torch


device = torch.device("cuda")
discrete_factor = 4


class Agent:
    def act(self, state, step):
        raise NotImplementedError()

    def save_table(self, reward, step):
        raise NotImplemented()


class EpsilonGreedyAgent(Agent):
    '''Эпсилон-жадный алгоритм. С вероятностью эпсилон совершает случайное действие, иначе - действие, которое хочет совершить другой агент'''

    def __init__(self, inner_agent, epsilon_max=0, epsilon_min=0, decay_steps=1000):
        self._inner_agent = inner_agent
        self._decay_steps = decay_steps
        self._steps = 0
        self._epsilon_max = epsilon_max
        self._epsilon_min = epsilon_min

    def act(self, state, step):
        self._steps = step
        epsilon = self._epsilon_max + (self._epsilon_min - self._epsilon_max) * self._steps / self._decay_steps
        if np.random.random() < epsilon:
            print('random')
            return np.random.randint(0, 2), [0, 0]
        else:
            return self._inner_agent.act(state, step)

    def save_table(self, reward, step):
        self._inner_agent.save_table()


class QLearning(Agent):
    def __init__(self, actions=2, alpha=0.1, gamma=0.9):
        self._table = np.zeros((600//discrete_factor, 600//discrete_factor, 100//discrete_factor, 2, actions), dtype=np.single)
        self._alpha = alpha
        self._gamma = gamma

    def update(self, transition):
        '''
        Обновление таблицы по одному переходу
        '''
        prev_state, action, state, reward, done = transition
        target_q = reward
        if not done:
            target_q += self._gamma * np.max(self._table[state[0]][state[1]][state[2]][state[3]])
        predicted_q = self._table[prev_state[0]][prev_state[1]][prev_state[2]][prev_state[3]][action]
        # print(target_q, predicted_q)
        self._table[prev_state[0]][prev_state[1]][prev_state[2]][prev_state[3]][action] =\
            (1 - self._alpha) * predicted_q + self._alpha * target_q

    def act(self, state, step):
        if np.sum(self._table[state[0]][state[1]][state[2]][state[3]] == 0) == 2:
            print('both zero')
        return np.argmax(self._table[state[0]][state[1]][state[2]][state[3]]), self._table[state[0]][state[1]][state[2]][state[3]]

    def save_table(self, reward, step):
        print()
        print('saving')
        non_zero = np.sum(self._table != 0)
        print('{} non-zero values'.format(non_zero))
        np.save('q_l_tables/Q_learning_{}_{}_{}.npy'.format(reward, non_zero, step), self._table)

    def load_table(self, name):
        self._table = np.load(name)


class QLearningUpdater:
    def __init__(self, qlearning: QLearning, backward=True):
        self._q_learning = qlearning
        self._backward = backward

    def update(self, trajectory):
        if self._backward:
            for transition in reversed(trajectory):
                self._q_learning.update(transition)
        else:
            for transition in trajectory:
                self._q_learning.update(transition)


def play_episode(env, agent: Agent, step):
    '''Играет один эпизод в заданной среде с заданным агентом'''
    print(f'epoch {step}')
    state = env.reset()
    state = [state[0]//discrete_factor, state[1]//discrete_factor, state[2]//discrete_factor, state[3]]
    done = False
    trajectory = []
    total_reward = 0
    final_rwd = 0
    while not done:
        action, table_values = agent.act(state, step)
        new_state, reward, done, info = env.step(action)
        new_state = [new_state[0]//discrete_factor, new_state[1]//discrete_factor, new_state[2]//discrete_factor, new_state[3]]
        total_reward += reward
        trajectory.append([state, action, new_state, reward, done])
        print('play ep: {}\t{}\t{}\t{}\t{}'.format(action, state, reward, done, table_values))
        state = new_state.copy()
        final_rwd = reward

    for i in range(len(trajectory)-1, max(0, len(trajectory) - 60), -1):
        final_rwd = final_rwd * 0.8
        trajectory[i][3] += final_rwd

    return trajectory, total_reward


def train_agent(env, exploration_agent: Agent, exploitation_agent: Agent, updater, total_epochs=1000, exploit_every=20):
    '''Обучает агента заданное количество шагов среды'''
    steps_count = 0
    exploits = 1
    rewards = []
    while steps_count <= total_epochs:
        if exploits * exploit_every <= steps_count:
            exploits += 1
            _, reward = play_episode(env, exploitation_agent, steps_count)
            rewards.append(reward)
            exploitation_agent.save_table(reward, steps_count)
        else:
            trajectory, reward = play_episode(env, exploration_agent, steps_count)
            steps_count += 1
            updater.update(trajectory)

    return rewards

