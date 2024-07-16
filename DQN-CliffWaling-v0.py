import torch
from torch import nn, optim, tensor

import gym
import numpy as np
import matplotlib.pyplot as plt

from random import random, sample
from collections import deque
from tqdm import tqdm

env = gym.make('CliffWalking-v0')
env_render = gym.make('CliffWalking-v0', render_mode='human')

RESET_PARAMETERS = True

EPISODES = 600

MAX_LEN = 100000

TAU = 0.005
BATCH_SIZE = 128

GAMMA = 0.99
LR = 0.1

EPS = 1.0
EPS_MIN = 0.0005
EPS_DECAY = EPS_MIN ** (1/EPISODES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.memory = deque([], maxlen=maxlen)
    
    def push(self, data: list):
        self.memory.append(data)
    
    def sample_data(self, batch_size: int):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.parameters(), lr=LR, amsgrad=True)

        self.to(device)

    def forward(self, X: tensor):
        pred = self.model(X)
        
        return pred
    
class Agent(Network):
    def __init__(self):
        super().__init__()

    def Action(self, obs: tensor):
        if EPS > random():
            action = env.action_space.sample()
            return action
        
        else:
            with torch.no_grad():
                action = self.forward(obs).argmax().item()
            return action
    
    def Education(self, Buffer, Q_target):
        if len(Buffer) < BATCH_SIZE:
            return 
            
        self.train()
        
        batch = Buffer.sample_data(BATCH_SIZE)

        obs, action, reward, next_obs = zip(*batch)

        obs, action, reward, next_obs = np.array(obs), np.array(action), np.array(reward), np.array(next_obs)
        obs, action, reward, next_obs = tensor(obs, dtype=torch.float32, device=device), tensor(action, dtype=torch.int64, device=device), tensor(reward, device=device), tensor(next_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            Q_value_of_next_states = Q_target(next_obs)
        Max_Q_values = Q_value_of_next_states.max(1)[0]
        
        Target_Q_value = reward + GAMMA * Max_Q_values
        Q_value = self.forward(obs).gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(Q_value, Target_Q_value)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()

        self.eval()
    
    def soft_update(self, Q_target, Q_policy):
        policy_state_dict = Q_policy.state_dict()
        target_state_dict = Q_target.state_dict()

        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key]*TAU + target_state_dict[key]*(1-TAU)
        Q_target.load_state_dict(target_state_dict)

    def Reset_parameters(self):
        for param in self.model.children():
            if hasattr(param, 'reset_parameters'):
                param.reset_parameters()

class Graphic:
    def __init__(self, x: str, y: str, title: str):
        plt.ion()

        plt.figure(1)

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title)

        self.list_of_episodes = []
        self.list_of_accuracy = []

    def update(self, episode: int, accuracy: int):
        self.list_of_episodes.append(episode)
        self.list_of_accuracy.append(accuracy)

        plt.clf()
        plt.plot(self.list_of_episodes, self.list_of_accuracy)
        plt.pause(0.001)
    
    def show(self, new_title: int):
        plt.ioff()

        plt.title(new_title)

        plt.clf()
        plt.plot(self.list_of_episodes, self.list_of_accuracy)

        plt.show()

def step(pbar):
    global EPS

    obs, info = env.reset()

    reward_per_episode = 0

    steps = 0
    while steps != 200:
        action = Agent.Action(tensor([obs], dtype=torch.float32, device=device))

        next_obs, reward, done, _, _ = env.step(action)

        #env.render()

        pbar.set_description(f"Obs: {obs} || Action: {action}, EPS: {EPS: .4f} || reward: {reward}, done: {done}")

        Buffer.push([[obs], action, reward, [next_obs]])

        if (steps+1) % 5 == 0:
            Agent.Education(Buffer, Q_target)
        
        if (steps+1) % 100 == 0:
            Agent.soft_update(Q_target, Q_policy)

        obs = next_obs

        reward_per_episode += reward
        steps += 1

        if done:
            break

    EPS = EPS * EPS_DECAY if EPS > EPS_MIN else EPS_MIN

    Graphic.update(episode, reward_per_episode)

Buffer = ReplayBuffer(MAX_LEN)

Q_policy = Network()
Q_target = Network()

Q_target.load_state_dict(Q_policy.state_dict())

Agent = Agent()

Graphic = Graphic(
    x='episodes',
    y='accuracy',
    title='In progress...'
)

if RESET_PARAMETERS:
    Agent.Reset_parameters()

    Q_target.load_state_dict(Q_policy.state_dict())

for episode in (pbar := tqdm(range(EPISODES))):
    step(pbar)

    if (episode+1) == (EPISODES-10):
        env = env_render

env.close()

Graphic.show(new_title='Result')