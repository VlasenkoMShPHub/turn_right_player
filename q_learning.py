import time
import copy
import random
import numpy as np
from collections import deque
# PyTorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


# Устройство, на котором будет работать PyTorch.
device = torch.device("cuda")
discrete_factor = 5


class Agent:
    '''Любой агент умеет возвращать действие по состоянию'''
    def act(self, state, step):
        raise NotImplementedError()

    def save_table(self, reward, step):
        raise NotImplemented()


class EpsilonGreedyAgent(Agent):
    '''Эпсилон-жадный алгоритм. С вероятностью эпсилон совершает случайное действие, иначе - действие, которое хочет совершить другой агент'''

    def __init__(self, inner_agent, epsilon_max=0.5, epsilon_min=0.1, decay_steps=1000):
        self._inner_agent = inner_agent
        self._decay_steps = decay_steps
        self._steps = 0
        self._epsilon_max = epsilon_max
        self._epsilon_min = epsilon_min

    def act(self, state, step):
        self._steps = step
        epsilon = self._epsilon_max + (self._epsilon_min - self._epsilon_max) * self._steps / self._decay_steps
        if epsilon > np.random.random():
            print('random')
            return np.random.randint(0, 2)
        else:
            return self._inner_agent.act(state, step)

    def save_table(self, reward, step):
        self._inner_agent.save_table()


class QLearning(Agent):
    def __init__(self, actions=2, alpha=0.5, gamma=0.8):
        self._table = np.zeros((int(5e7)//discrete_factor, actions), dtype=np.single)
        self._alpha = alpha
        self._gamma = gamma

    def update(self, transition):
        '''
        Обновление таблицы по одному переходу
        '''
        prev_state, action, state, reward, done = transition
        target_q = reward
        if not done:
            target_q += self._gamma * np.max(self._table[state])
        #print('target_q = ', target_q, done)
        predicted_q = self._table[prev_state][action]
        self._table[prev_state][action] = (1 - self._alpha) * predicted_q + self._alpha * target_q
        # print('result = ', self._table[prev_state][action])

    def act(self, state, step):
        return np.argmax(self._table[state])

    def save_table(self, reward, step):
        print()
        print('saving')
        non_zero = np.sum(self._table != 0)
        print('{} non-zero values'.format(non_zero))
        np.save('q_l_tables/Q_learning_{}_{}_{}.npy'.format(reward, non_zero, step), self._table)


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
    state = int(str(state[3]//discrete_factor) + str(state[2]//discrete_factor) + str(state[4]))
    done = False
    trajectory = []
    total_reward = 0
    while not done:
        action = agent.act(state, step)
        new_state, reward, done, info = env.step(action)
        new_state = int(str(new_state[3]//discrete_factor) + str(new_state[2]//discrete_factor) + str(new_state[4]))
        total_reward += reward
        trajectory.append((state, action, new_state, reward, done))
        print('play ep: {}\t{}\t{}\t{}'.format(action, state, reward, done))
        state = new_state
    return trajectory, total_reward


def train_agent(env, exploration_agent: Agent, exploitation_agent: Agent, updater, env_steps=1000, exploit_every=20):
    '''Обучает агента заданное количество шагов среды'''
    steps_count = 0
    exploits = 1
    rewards = []
    steps = []
    while steps_count < env_steps:
        if exploits * exploit_every <= steps_count:
            exploits += 1
            _, reward = play_episode(env, exploitation_agent, steps_count)
            rewards.append(reward)
            steps.append(steps_count)
            exploitation_agent.save_table(reward, steps_count)
        else:
            trajectory, reward = play_episode(env, exploration_agent, steps_count)
            steps_count += 1  # len(trajectory)
            updater.update(trajectory)

    _, reward = play_episode(env, exploitation_agent, env_steps)
    rewards.append(reward)
    steps.append(steps_count)
    return rewards, steps

