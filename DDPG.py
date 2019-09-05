import gym
import os
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm as tqdm
import random
import numpy as np
from collections import deque
import torch
import contextlib
from torch import nn
from torch import optim
import torch.nn.functional as F


ENV_ID = "BipedalWalker-v2"

if ENV_ID == "MountainCarContinuous-v0":
    GAMMA = 0.99
    TAU = 0.05
    BATCH_SIZE = 256
    ACTOR_LR = 1e-5
    CRITIC_LR = 1e-5
    MAX_EPISODES = 35
    MAX_TIMESTAMPS = 500
    SIGMA = 0.6
    EPS_MIN = 0.1  # from [0...1]
    ALGORITHM = 'DDPG'  # 'DDPG' or 'TD3'

elif ENV_ID == "LunarLanderContinuous-v2":
    GAMMA = 0.99
    TAU = 0.005
    BATCH_SIZE = 256
    ACTOR_LR = 1e-5
    CRITIC_LR = 1e-4
    MAX_EPISODES = 150
    MAX_TIMESTAMPS = 500
    SIGMA = 0.4
    EPS_MIN = 0.1  # from [0...1]
    ALGORITHM = 'DDPG'  # 'DDPG' or 'TD3'

elif ENV_ID == "BipedalWalker-v2":
    GAMMA = 0.99
    TAU = 0.05
    BATCH_SIZE = 256
    ACTOR_LR = 1e-3
    CRITIC_LR = 5e-4
    MAX_EPISODES = 300
    MAX_TIMESTAMPS = 500
    SIGMA = 0.6
    EPS_MIN = 0.2  # from [0...1]
    ALGORITHM = 'TD3'  # 'DDPG' or 'TD3'
else:
    raise Exception(f'Unknown ENV_ID: {ENV_ID}')

CAPACITY = 50000
EXPLOIT_INTERVAL = MAX_EPISODES // 50 if MAX_EPISODES >= 50 else 1
EXPLOIT_EPISODES = 3
RERUN_EPISODES = 30
SEED = 1234
DEVICE = 'gpu'  # 'gpu' or 'cpu'
TD3_POLICY_DELAY = 2

if not torch.cuda.is_available() and DEVICE == 'gpu':
    print('No gpu detected, using cpu...')
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device("cuda:0" if DEVICE == 'gpu' else "cpu")
if not os.path.exists('./weights'):
    os.mkdir('weights')
print(f'device = {DEVICE}')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def to_torch(x, dtype=torch.float):
    return torch.tensor(x, dtype=dtype, device=DEVICE)


def to_numpy(x):
    return x.detach().cpu().numpy()


class OUNoise:
    def __init__(self, action_space):
        self.theta = 0.15
        self.sigma = SIGMA
        self.action_dim = action_space.shape[0]
        self.low = action_space.low[0]
        self.high = action_space.high[0]
        self.eps_max = 1.0
        self.eps_min = EPS_MIN
        self.eps = self.eps_max
        self.eps_decay = (self.eps_max - self.eps_min) / MAX_EPISODES
        self.state = np.zeros(self.action_dim)

    def eps_step(self):
        self.eps -= self.eps_decay

    def evolve_state(self):
        self.state += - self.theta * self.state + self.sigma * np.random.randn(self.action_dim)

    def __call__(self, action):
        self.evolve_state()
        noise = action + self.state * self.eps
        return np.clip(noise, self.low, self.high)


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

    def make_action(self, state):
        state = to_torch(state)
        action = self.forward(state)
        return to_numpy(action)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state, action):
        obs = self.obs_net(state)
        return self.out_net(torch.cat([obs, action], dim=1))


def ddpg_update(actor, critic, replay_buffer):
    # sample from memory
    state, action, reward, next_state, done = \
        [to_torch(e) for e in zip(*random.sample(replay_buffer, BATCH_SIZE))]

    """
    YOUR CODE GOES HERE
    """
    # get qvalue + target qvalue
    expected_qvalue = critic.target(next_state, actor.target(next_state))
    expected_qvalue = reward.unsqueeze(1) + (1.0 - done.unsqueeze(1)) * GAMMA * expected_qvalue
    expected_qvalue = expected_qvalue.detach()
    qvalue = critic(state, action)

    # update critic
    critic_loss = F.mse_loss(qvalue, expected_qvalue)
    critic.optimizer.zero_grad()
    critic_loss.backward()
    critic.optimizer.step()

    # update actor
    actor_loss = - critic(state, actor(state)).mean()
    actor.optimizer.zero_grad()
    actor_loss.backward()
    actor.optimizer.step()

    # update target networks
    for target_param, param in zip(critic.target.parameters(), critic.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
    for target_param, param in zip(actor.target.parameters(), actor.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

    return actor, critic, replay_buffer


def exploit_episode(env, actor_tgt, render=False):
    state = env.reset()
    done = 0
    ep_reward = 0
    """
    YOUR CODE GOES HERE
    """
    while not done:
        if render:
            env.render()
        action = actor.make_action(state)
        np.clip(action, env.action_space.low, env.action_space.high)
        state, reward, done, _ = env.step(action)
        ep_reward += reward
    return ep_reward


def exploit(env, actor_tgt, num, render=False):
    return [exploit_episode(env, actor_tgt, render) for _ in range(num)]


def play_episode(noise, actor, critic, replay_buffer):
    state = env.reset()
    noise.eps_step()
    ep_reward = 0

    """
    YOUR CODE GOES HERE
    """
    for i in range(MAX_TIMESTAMPS):
        action = actor.make_action(state)
        print(f'action = {action}')
        action = noise(action)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        ep_reward += reward
        state = next_state
        if len(replay_buffer) > BATCH_SIZE:
            actor, critic, replay_buffer = ddpg_update(actor, critic, replay_buffer)

        if done:
            break

    return ep_reward, noise, actor, critic, replay_buffer


def save(id_episode, actor_tgt, reward):
    weights_name = f"{ENV_ID}_ep={id_episode}_r={reward:.2f}"
    torch.save(actor_tgt.state_dict(), f"weights/{weights_name}")
    return weights_name


def load(actor_tgt, weights_name):
    state_dict = torch.load(f"weights/{weights_name}", map_location=lambda storage, location: storage)
    actor_tgt.load_state_dict(state_dict)
    return actor_tgt


def rerun_best(weights_name):
    env_name = weights_name.split('_')[0]
    with contextlib.redirect_stdout(None):
        env = gym.make(env_name)
    actor_tgt = DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    actor_tgt = load(actor_tgt, weights_name)
    exploit_rewards = exploit(env, actor_tgt, num=RERUN_EPISODES)
    print(f"Rerunning best target model {RERUN_EPISODES} times: mean_r={np.mean(exploit_rewards):.3f}")


def show_plots(train_rewards, exploit_rewards):
    train_times = [x for x in range(1, MAX_EPISODES + 1)]
    exploit_times = [x for x in range(1, MAX_EPISODES + 1) if x % EXPLOIT_INTERVAL == 0]
    plt.figure(figsize=(10, 8))
    plt.plot(train_times, train_rewards, label='Train Rewards')
    plt.plot(exploit_times, exploit_rewards, label='Exploit Rewards')
    plt.legend(loc=4, prop={'size': 18})
    plt.show()


def get_tools():
    with contextlib.redirect_stdout(None):
        env = gym.make(ENV_ID)
        env.seed(SEED)
    print(f"{ENV_ID}\n"
          f"observation space: {env.observation_space}\n"
          f"action space: {env.action_space}\n"
          f"action range: ({env.action_space.low[0]}, {env.action_space.high[0]})\n")

    # define noise
    noise = OUNoise(env.action_space)

    # define actor
    actor_local = DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    actor_target = copy.deepcopy(actor_local)
    actor_optimizer = optim.Adam(actor_local.parameters(), lr=ACTOR_LR)
    actor = actor_local
    actor.target = actor_target
    actor.optimizer = actor_optimizer

    if ALGORITHM == 'DDPG':
        # define critic
        critic_local = DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
        critic_target = copy.deepcopy(critic_local)
        critic_optimizer = optim.Adam(critic_local.parameters(), lr=CRITIC_LR)
        critic = critic_local
        critic.target = critic_target
        critic.optimizer = critic_optimizer

    if ALGORITHM == 'TD3':
        # define critic1
        critic_local1 = DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
        critic_target1 = copy.deepcopy(critic_local1)
        critic_optimizer1 = optim.Adam(critic_local1.parameters(), lr=CRITIC_LR)
        critic1 = critic_local1
        critic1.target = critic_target1
        critic1.optimizer = critic_optimizer1

        # define critic2
        critic_local2 = DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
        critic_target2 = copy.deepcopy(critic_local2)
        critic_optimizer2 = optim.Adam(critic_local2.parameters(), lr=CRITIC_LR)
        critic2 = critic_local2
        critic2.target = critic_target2
        critic2.optimizer = critic_optimizer2
        critic = (critic1, critic2)

    # define replay buffer
    replay_buffer = deque(maxlen=CAPACITY)

    return env, noise, actor, critic, replay_buffer


def train_draft(env, noise, actor, critic, replay_buffer):
    train_rewards = []
    exploit_rewards = []
    best_episode_weights = None
    best_exploit_reward = -np.inf

    pbar = tqdm(range(1, MAX_EPISODES + 1), total=MAX_EPISODES)
    for id_episode in pbar:

        # play + train episode
        ep_reward, noise, actor, critic, replay_buffer = \
            play_episode(noise, actor, critic, replay_buffer)

        # record episode reward
        train_rewards.append(ep_reward)
        pbar.set_description(f"R: {train_rewards[-1]:.3f}")

        # evaluate by exploitation
        """
        YOUR CODE GOES HERE
        """
        if id_episode % EXPLOIT_INTERVAL == 0:
            exploit_rewards.append(np.mean(exploit(env, actor.target, EXPLOIT_EPISODES)))
            print(f"episode: {id_episode} | exploit reward: {exploit_rewards[-1]:.3f}")
            if best_exploit_reward < exploit_rewards[-1]:
                best_exploit_reward = exploit_rewards[-1]
                best_episode_weights = save(id_episode, actor.target, best_exploit_reward)

    print(f"Weights for best episode: {best_episode_weights}")
    return train_rewards, exploit_rewards, best_episode_weights


env, noise, actor, critic, replay_buffer = get_tools()
train_rewards, exploit_rewards, best_weights = train_draft(env, noise, actor, critic, replay_buffer)
rerun_best(best_weights)
