import time
import numpy as np
import torch

device = torch.device("cuda")
discrete_factor = 3


class Agent:
    def act(self, state):
        raise NotImplementedError()

    def save_table(self, reward, step):
        raise NotImplemented()


class EpsilonGreedyAgent(Agent):
    def __init__(self, inner_agent, epsilon_max=0.5, epsilon_min=0.1, decay_steps=1000000):
        self._inner_agent = inner_agent
        self._decay_steps = decay_steps
        self._steps = 0
        self._epsilon_max = epsilon_max
        self._epsilon_min = epsilon_min

    def act(self, state):
        self._steps = min(self._decay_steps, self._steps + 1)
        epsilon = self._epsilon_max + (self._epsilon_min - self._epsilon_max) * self._steps / self._decay_steps
        if epsilon > np.random.random():
            return np.random.randint(0, 2)
        else:
            return self._inner_agent.act(state)

    def save_table(self, reward, step):
        self._inner_agent.save_table()


class QLearning(Agent):
    def __init__(self, actions=2, alpha=0.9, gamma=0.99):
        self._table = np.zeros((600 // discrete_factor, 600 // discrete_factor, 100 // discrete_factor, 2, actions),
                               dtype=np.float32)
        self._alpha = alpha
        self._gamma = gamma

    def update(self, transition):
        '''Обновление таблицы по одному переходу'''
        prev_state, action, state, reward, done = transition
        target_q = reward
        if not done:
            target_q += self._gamma * np.max(self._table[state[0]][state[1]][state[2]][state[3]])

        self._table[prev_state[0]][prev_state[1]][prev_state[2]][prev_state[3]][action] += \
            self._alpha * (target_q - self._table[prev_state[0]][prev_state[1]][prev_state[2]][prev_state[3]][action])
        # print(target_q, self._table[prev_state[0]][prev_state[1]][prev_state[2]][prev_state[3]][action])

    def act(self, state):
        if np.sum(self._table[state[0]][state[1]][state[2]][state[3]] == 0) == 2:
            print('both zero')
        else:
            print(self._table[state[0]][state[1]][state[2]][state[3]])
        return np.argmax(self._table[state[0]][state[1]][state[2]][state[3]])

    def save_table(self, reward, step):
        print()
        print('saving')
        non_zero = np.sum(self._table != 0)
        print('{} non-zero values'.format(non_zero))
        np.save('q_l_tables/Q_learning_{}_{}_{}.npy'.format(reward, non_zero, step), self._table)

    def load_table(self, name):
        self._table = np.load(name)


class QLearningUpdater:
    def __init__(self, q_learning: QLearning):
        self._q_learning = q_learning
        self._backward = True

    def update(self, trajectory):
        print(trajectory)
        if self._backward:
            for transition in reversed(trajectory):
                self._q_learning.update(transition)
        else:
            for transition in trajectory:
                self._q_learning.update(transition)


def play_episode(env, agent: Agent, render=False):
    '''Играет один эпизод в заданной среде с заданным агентом'''
    state = env.reset()
    state = [state[0] // discrete_factor, state[1] // discrete_factor, state[2] // discrete_factor, state[3]]
    done = False
    trajectory = []
    total_reward = 0
    while not done:
        if render:
            env.render()
            time.sleep(0.01)
        action = agent.act(state)
        new_state, reward, done, info = env.step(action)
        new_state = [new_state[0] // discrete_factor, new_state[1] // discrete_factor,
                     new_state[2] // discrete_factor, new_state[3]]
        total_reward += reward
        trajectory.append((state, action, new_state, reward, done))
        print('play ep: {}\t{}\t{}\t{}'.format(action, state, reward, done))
        state = new_state.copy()
    return trajectory, total_reward


def train_agent(env, exploration_agent: Agent, exploitation_agent: Agent, updater, env_steps=600000,
                exploit_every=3000):
    '''Обучает агента заданное количество шагов среды'''
    steps_count = 0
    exploits = 1
    rewards = []
    steps = []
    while steps_count < env_steps:
        if exploits * exploit_every <= steps_count:
            exploits += 1
            _, reward = play_episode(env, exploitation_agent)
            rewards.append(reward)
            steps.append(steps_count)
            exploitation_agent.save_table(reward, steps_count)
        else:
            trajectory, reward = play_episode(env, exploration_agent)
            steps_count += len(trajectory)
            updater.update(trajectory)

    _, reward = play_episode(env, exploitation_agent)
    rewards.append(reward)
    steps.append(steps_count)
    return rewards, steps
