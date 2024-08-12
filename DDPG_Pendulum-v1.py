import torch
from torch import nn, optim, tensor
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import gym
from random import sample
from tqdm import tqdm

env = gym.make('Pendulum-v1')
env_render = gym.make('Pendulum-v1', render_mode='human')

EPISODES = 200

BATCH_SIZE = 128
MAXLEN = 100000

TAU = 0.005
GAMMA = 0.99

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.memory = deque([], maxlen=maxlen)
    
    def push(self, data: list):
        self.memory.append(data)

    def sample_batch(self, batch_size: int):
        return sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class ActorNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
        self.to(device)

    def forward(self, obs: tensor):
        pred = self.model(obs)

        return pred

class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.to(device)
    
    def forward(self, obs: tensor, action: tensor):
        obs = torch.cat([obs, action], dim=1)
        pred = self.model(obs)

        return pred

class DDPG:
    def __init__(self):

        self.actor_optimizer = optim.AdamW(Actor.parameters(), lr=0.0005, amsgrad=True)
        self.critic_optimizer = optim.AdamW(Critic.parameters(), lr=0.005, amsgrad=True)

        self.loss_fn = nn.SmoothL1Loss()

        self.mu = 0.0
        self.theta = 0.15
        self.sigma = 0.2
        self.current_noise = 0.0
    
    def action(self, obs: tensor):
        with torch.no_grad():
            action = Actor(obs)
        noise = self.OUNoise()

        action += noise

        return np.clip(np.array([action.item()]), -2, 2)
    
    def OUNoise(self):
        dx = self.theta * (self.mu - self.current_noise) + self.sigma * np.random.randn()
        self.current_noise += dx

        return self.current_noise
    
    def education(self):
        if len(Buffer) < BATCH_SIZE:
            return 
        
        batch = Buffer.sample_batch(BATCH_SIZE)
        obs, action, reward, next_obs, done = zip(*batch)
        
        obs = tensor(np.array(obs), dtype=torch.float32, device=device)
        action = tensor(np.array(action), dtype=torch.float32, device=device)
        reward = tensor(np.array(reward), dtype=torch.float32, device=device)
        next_obs = tensor(np.array(next_obs), dtype=torch.float32, device=device)
        
        done = tensor(np.array([i for i in map(lambda x: 1-int(x), done)]), device=device)

        with torch.no_grad():
            next_action = Actor_target(next_obs)
            target_q_value = Critic_target(next_obs, next_action).squeeze(1)
            target_q_value = reward + GAMMA * target_q_value * done
        
        current_q_value = Critic(obs, action).squeeze(1)
        critic_loss = self.loss_fn(current_q_value, target_q_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        predicted_action = Actor(obs)
        actor_loss = -Critic(obs, predicted_action).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def soft_update(self):
        Actor_state_dict = Actor.state_dict()
        Critic_state_dict = Critic.state_dict()
        Actor_target_state_dict = Actor_target.state_dict()
        Critic_target_state_dict = Critic_target.state_dict()

        for key in Actor_state_dict:
            Actor_target_state_dict[key] = Actor_state_dict[key]*TAU + Actor_target_state_dict[key]*(1-TAU)
        Actor_target.load_state_dict(Actor_target_state_dict)

        for key in Critic_state_dict:
            Critic_target_state_dict[key] = Critic_state_dict[key]*TAU + Critic_target_state_dict[key]*(1-TAU)
        Critic_target.load_state_dict(Critic_target_state_dict)

class Graphic:
    def __init__(self, x: str, y: str, title: str):
        plt.ion()
        
        self.x = x
        self.y = y
        self.title = title

        self.episodes = []
        self.rewards = []

    def update(self, episode, reward):
        self.episodes.append(episode)
        self.rewards.append(reward)

        plt.clf()
        plt.plot(self.episodes, self.rewards)
        plt.draw()
        plt.pause(0.05)

        plt.xlabel(self.x)
        plt.ylabel(self.y)
        plt.title(self.title)
    
    def show(self):
        plt.ioff()
        
        plt.clf()
        plt.plot(self.episodes, self.rewards)
        plt.pause(0.05)

        plt.xlabel(self.x)
        plt.ylabel(self.y)
        plt.title(self.title)

        plt.show()

def step(pbar, episode):
    obs, info = env.reset()

    DDPG.current_noise = 0.0

    total_reward = 0
    truncated = False
    while True:
        action = DDPG.action(tensor(obs, dtype=torch.float32, device=device))

        next_obs, reward, done, truncated, _ = env.step(action)

        pbar.set_description(f"action: {action.item(): .4f} | reward: {reward: .4f}, {truncated}")

        Buffer.push([obs, action, reward, next_obs, truncated])

        if (episode+1) % 4 == 0:
            DDPG.education()

        if (episode+1) % 20 == 0:
            DDPG.soft_update()

        obs = next_obs
        total_reward += reward

        if truncated:
            break
    
    Graphic.update(episode, total_reward)

Buffer = ReplayBuffer(MAXLEN)

Actor = ActorNetwork()
Actor_target = ActorNetwork()
Critic = CriticNetwork()
Critic_target = CriticNetwork()

DDPG = DDPG()

Graphic = Graphic(
    x='episode',
    y='reward',
    title='DDPG Pendulum-v1'
)

Actor_target.load_state_dict(Actor.state_dict())
Critic_target.load_state_dict(Critic.state_dict())

for episode in (pbar := tqdm(range(EPISODES))):
    step(pbar, episode)

    if (episode+1) == (EPISODES-10):
        env = env_render

env.close()

Graphic.show()
