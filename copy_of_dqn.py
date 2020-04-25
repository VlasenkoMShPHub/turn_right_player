import time
import matplotlib.pyplot as plt
import copy
import random
import numpy as np
from collections import deque
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from input import Env


device = torch.device("cpu")


class Agent:
    def act(self, state):
        raise NotImplementedError()


class RandomAgent(Agent):
    def act(self, state):
        return np.random.randint(0, 3)


class EpsilonGreedyAgent(Agent):
    '''Эпсилон-жадный алгоритм. С вероятностью эпсилон совершает случайное действие, иначе - действие, которое хочет совершить другой агент'''

    def __init__(self, inner_agent, epsilon_max=0.75, epsilon_min=0.1, decay_steps=2000000):
        self._inner_agent = inner_agent
        self._decay_steps = decay_steps
        self._steps = 0
        self._epsilon_max = epsilon_max
        self._epsilon_min = epsilon_min

    def act(self, state):
        self._steps = min(self._decay_steps, self._steps + 1)
        epsilon = self._epsilon_max + (self._epsilon_min - self._epsilon_max) * self._steps / self._decay_steps
        if epsilon > np.random.random():
            return np.random.random(1) * 2 - 1
        else:
            return self._inner_agent.act(state)


class Updater:
    '''Класс, который отвечает за обновление агента'''
    def update(self, trajectory):
        pass


def play_episode(env, agent: Agent, render=False):
    '''Играет один эпизод в заданной среде с заданным агентом'''
    state = env.reset()
    done = False
    trajectory = []
    total_reward = 0
    info = dict()
    while not done:
        if render:
            env.render()
            time.sleep(0.01)
        action = agent.act(state)
        new_state, reward, done, info = env.step(action)
        total_reward += reward
        trajectory.append((state, action, new_state, reward, done))
        state = new_state
        print('play ep: {}\t{}\t{}\t{}'.format(action, state, reward, done))
    print(f'ep_reward = {total_reward}, time = {info["time"]}')
    print()
    return trajectory, total_reward


def train_agent(env, exploration_agent: Agent, exploitation_agent: Agent, updater: Updater, env_steps=2000000,
                exploit_every=20000):
    '''Обучает агента заданное количество шагов среды'''
    steps_count = 0
    exploits = 0
    rewards = []
    steps = []
    while steps_count < env_steps:
        if exploits * exploit_every <= steps_count:
            exploits += 1
            _, reward = play_episode(env, exploitation_agent)
            rewards.append(reward)
            steps.append(steps_count)
        else:
            trajectory, reward = play_episode(env, exploration_agent)
            steps_count += len(trajectory)
            updater.update(trajectory)

    _, reward = play_episode(env, exploitation_agent)
    rewards.append(reward)
    steps.append(steps_count)
    return rewards, steps


"""### DQN

Теперь попробуем решить эту же задачу при помощи DQN. Теперь состояния непрерывны, а агентом является нейронная сеть.

Опишем самого агента DQN, а так же Updater. Стоит отметить, что в отличии от Q-learning, мы теперь обучаем не на кокретном переходе среды, а на батче, собранном из ранее наблюдаемых переходов.
"""


class DQN(Agent):
    def __init__(self, gamma=0.99, tau=0.01, learning_rate=0.0001, device=torch.device("cpu")):
        self._device = device
        self._model = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.shape[0]),
            nn.Tanh(),
        )

        self._target_model = copy.deepcopy(self._model)

        # Инициализация весов нейронной сети
        def init_weights(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal(layer.weight)

        self._model.apply(init_weights)

        # Загружаем модель на устройство, определенное в самом начале (GPU или CPU)
        self._model.train()
        self._target_model.eval()
        self._model.to(self._device)
        self._target_model.to(self._device)

        # Сразу зададим оптимизатор, с помощью которого будем обновлять веса модели
        self._optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)
        self._gamma = gamma
        self._tau = tau

    def act(self, state):
        state = torch.tensor(state).to(self._device).float()
        # print(self._target_model(state))
        action = self._target_model(state).detach().cpu().numpy()
        print(action)
        return action

    def update(self, batch):
        prev_states, actions, states, rewards, dones = batch
        # Загружаем батч на выбранное ранее устройство
        prev_states = torch.tensor(prev_states).to(self._device).float()
        states = torch.tensor(states).to(self._device).float()
        rewards = torch.tensor(rewards).to(self._device).float()
        actions = torch.tensor(actions).to(self._device)

        # Считаем то, какие значения должна выдавать наша сеть
        target_q = torch.zeros(rewards.size()[0]).float().to(self._device)
        with torch.no_grad():
            # Выбираем максимальное из значений Q-function для следующего состояяния
            mask = [not d for d in dones]
            target_q[mask] = self._target_model(states[mask]).max(1)[0]
        target_q = rewards + target_q * self._gamma

        # Current approximation
        q = self._model(prev_states).gather(0, actions.unsqueeze(1)).view(-1)  # used to be gather(1,

        q.to(self._device)
        loss = F.smooth_l1_loss(q, target_q)

        # Очищаем текущие градиенты внутри сети
        self._optimizer.zero_grad()
        # Применяем обратное распространение  ошибки
        loss.backward()
        # Делаем шаг оптимизации
        self._optimizer.step()

        # Soft-update
        for target_param, param in zip(self._target_model.parameters(), self._model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self._tau) + param.data * self._tau)

        return torch.abs(q - target_q).detach().cpu().numpy()


class DQNUpdater(Updater):
    def __init__(self, dqn, buffer_size=5000, batch_size=128):
        self._memory_buffer = deque(maxlen=buffer_size)
        self._dqn = dqn
        self._batch_size = batch_size

    def update(self, trajectory):
        # start_time = time.time()
        for transition in trajectory:
            self._memory_buffer.append(transition)
            if len(self._memory_buffer) > self._batch_size:
                batch = random.sample(self._memory_buffer, self._batch_size)
                batch = list(zip(*batch))
                self._dqn.update(batch)
        # print("Trajectory time:", time.time() - start_time)


"""### Prioretized Expirience Replay
Теперь попробуем ускорить обучение при помощи более эффективного использования имеющихся у нас данных.
Для этого реализуем структуру, позволяющую быстро обновлять приоритеты для переходов среды и соответствующий ей Updater.
"""


class PriorityTree:
    def __init__(self, capacity, alpha=0.6):
        self._leaves = [(None, 0)] * capacity
        self._tree = [0] * (2 * capacity - 1)
        self._i = 0
        self._alpha = alpha

    def add(self, transition, error):
        self._leaves[self._i] = (transition, error ** self._alpha)
        self._update(self._i)
        self._i = (self._i + 1) % len(self._leaves)

    def refresh(self, i, error):
        self._leaves[i] = (self._leaves[i][0], error ** self._alpha)
        self._update(i)

    def _update(self, i):
        self._tree[len(self._leaves) - 1 + i] = self._leaves[self._i][1]
        k = (len(self._leaves) - 1 + i - 1) // 2
        '''
        Подъем по дереву
        '''
        while k != 0:
            self._tree[k] = self._tree[2 * k + 1] + self._tree[2 * k + 2]
            k = (k - 1) // 2
        self._tree[0] = self._tree[1] + self._tree[2]

    def sample(self, batch_size):
        ids = []
        batch = []
        for _ in range(batch_size):
            i, t = self.sample_single()
            ids.append(i)
            batch.append(t)
        return ids, batch

    def sample_single(self):
        rnd = random.random()
        rnd *= self._tree[0]
        brnd = rnd
        k = 0
        """
        Спуск по дереву
        """
        while k < len(self._leaves) - 1:
            if self._tree[2 * k + 1] < rnd:
                rnd -= self._tree[2 * k + 1]
                k = 2 * k + 2
            else:
                k = 2 * k + 1
        i = k - (len(self._leaves) - 1)
        return i, self._leaves[i][0]


class PriorityDQNUpdater(Updater):
    def __init__(self, dqn, buffer_size=5000, batch_size=128, alpha=0.6):
        self._memory_buffer = PriorityTree(buffer_size, alpha)
        self._not_ranked_transitions = []
        self._dqn = dqn
        self._batch_size = batch_size

    def update(self, trajectory):
        for transition in trajectory:
            if self._not_ranked_transitions is not None:
                self._not_ranked_transitions.append(transition)
                if len(self._not_ranked_transitions) == self._batch_size:
                    batch = list(zip(*self._not_ranked_transitions))
                    errors = self._dqn.update(batch)
                    for t, e in zip(self._not_ranked_transitions, errors):
                        self._memory_buffer.add(t, e)
                    self._not_ranked_transitions = None
            else:
                ids, batch = self._memory_buffer.sample(self._batch_size - 1)
                batch.append(transition)
                batch = list(zip(*batch))
                errors = self._dqn.update(batch)
                for i, e in zip(ids, errors):
                    self._memory_buffer.refresh(i, e)
                self._memory_buffer.add(transition, errors[-1])


max_steps = 50000

env = Env()
dqn_with_per_agent = DQN(device=device)
updater = PriorityDQNUpdater(dqn_with_per_agent, buffer_size=65536)  # Для корректной работы должен быть степенью 2
epsilon_greedy = EpsilonGreedyAgent(dqn_with_per_agent, decay_steps=max_steps)
start_time = time.time()
rewards, steps = train_agent(env, epsilon_greedy, dqn_with_per_agent, updater, env_steps=max_steps,
                             exploit_every=10)
# print("Training time:", time.time() - start_time)

# plot_rewards(rewards, steps)
