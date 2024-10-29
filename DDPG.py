import torch
from torch import tensor, from_numpy, nn, optim, device, cuda
import torch.nn.functional as F

import numpy as np

from random import sample
from collections import deque

device = device('cuda' if cuda.is_available() else 'cpu')

class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.memory = deque([], maxlen=maxlen)
    
    def push(self, data: list):
        self.memory.append([np.array(x) for x in data])
    
    def sample_batch(self, batch_size: int):
        return sample(self.memory, k=batch_size)
    
    def __len__(self):
        return len(self.memory)

class ActorCritic(nn.Module):
    def __init__(self, action_dim: int, observ_dim: int, action_scaling: float):
        super().__init__()

        self.action_scaling = action_scaling

        self.Actor = nn.Sequential(
            nn.Linear(observ_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim)
        )

        self.Critic = nn.Sequential(
            nn.Linear(observ_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

        for layer in self.Actor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

        for layer in self.Critic:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

        self.to(device)
    
    def get_action(self, state: tensor):
        pred = F.tanh(self.Actor(state)) * self.action_scaling

        return pred
    
    def get_value_of_action(self, state: tensor, action: tensor):
        input_data = torch.cat([state, action], dim=1)
        pred = self.Critic(input_data)

        return pred

class DDPG:
    def __init__(self, max_action: float, min_action: float, Action_dim: int, Observ_dim: int, Actor_lr: float = 0.0003, Critic_lr: float = 0.0025, batch_size: int = 64, gamma: float = 0.98, TAU: float = 0.0005, max_len_of_buffer: int = 50000, mu: float = 0.0, sigma: float = 0.2, theta: float = 0.15):
        self.policy = ActorCritic(Action_dim, Observ_dim, max_action)
        self.policy_target = ActorCritic(Action_dim, Observ_dim, max_action)

        self.Buffer = ReplayBuffer(max_len_of_buffer)

        self.policy_target.Actor.load_state_dict(self.policy.Actor.state_dict())
        self.policy_target.Critic.load_state_dict(self.policy.Critic.state_dict())

        self.loss_fn = nn.SmoothL1Loss()
        self.Actor_optimizer = optim.AdamW(self.policy.Actor.parameters(), lr=Actor_lr, amsgrad=True)
        self.Critic_optimizer = optim.AdamW(self.policy.Critic.parameters(), lr=Critic_lr, amsgrad=True)

        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.current_noise = np.full(Action_dim, mu)

        self.batch_size = batch_size
        self.gamma = gamma
        self.TAU = TAU
        self.Actor_lr = Actor_lr
        self.Critic_lr = Critic_lr
        self.max_len_of_buffer = max_len_of_buffer

        self.max_action = max_action
        self.min_action = min_action

        self.action_dim = Action_dim
        self.observ_dim = Observ_dim
    
    def get_action(self, state):
        with torch.no_grad():
            state = tensor(state, dtype=torch.float32, device=device)

            action = self.policy.get_action(state)
        noise = self.OUNoise()

        action += noise

        return np.clip(np.array([a.item() for a in action]), self.min_action, self.max_action)
    
    def OUNoise(self):
        dx = self.theta * (self.mu - self.current_noise) + self.sigma * np.random.randn(self.action_dim)
        self.current_noise += dx

        return from_numpy(self.current_noise).to(device)

    def reset_OUNoise(self):
        self.current_noise = np.full(self.action_dim, self.mu)

    def education(self):
        if len(self.Buffer) < self.batch_size:
            return 

        batch = self.Buffer.sample_batch(self.batch_size)
        
        states, actions, rewards, next_states = zip(*batch)

        states = tensor(np.array(states), dtype=torch.float32, device=device)
        actions = tensor(np.array(actions), dtype=torch.float32, device=device)
        rewards = tensor(np.array(rewards), dtype=torch.float32, device=device)
        next_states = tensor(np.array(next_states), dtype=torch.float32, device=device)

        with torch.no_grad():
            # Генерация действий для следующего состояния с использованием целевой политики
            target_actions = self.policy_target.get_action(next_states)
            
            # Получение Q-значений для следующих состояний и сгенерированных действий
            target_value_of_actions = self.policy_target.get_value_of_action(next_states, target_actions).squeeze(1)

            # Обновление целевого Q-значения
            target_Q_value = rewards + self.gamma * target_value_of_actions

        # Получение текущих Q-значений для текущих состояний и действий
        current_q_value = self.policy.get_value_of_action(states, actions).squeeze(1)
        critic_loss = self.loss_fn(current_q_value, target_Q_value)

        # Обновление градиентов и шаг оптимизации
        self.Critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_value_(self.policy.Critic.parameters(), 0.5)
        self.Critic_optimizer.step()
        
        predicted_actions = self.policy.get_action(states)
        actor_loss = -self.policy.get_value_of_action(states, predicted_actions).mean()
        
        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_value_(self.policy.Actor.parameters(), 0.5)
        self.Actor_optimizer.step()
    
    def soft_update(self):
        Actor_state_dict = self.policy.Actor.state_dict()
        Critic_state_dict = self.policy.Critic.state_dict()
        Actor_target_state_dict = self.policy_target.Actor.state_dict()
        Critic_target_state_dict = self.policy_target.Critic.state_dict()

        for key in Actor_state_dict:
            Actor_target_state_dict[key] = Actor_state_dict[key] * self.TAU + Actor_target_state_dict[key] * (1 - self.TAU)
        self.policy_target.Actor.load_state_dict(Actor_target_state_dict)

        for key in Critic_state_dict:
            Critic_target_state_dict[key] = Critic_state_dict[key] * self.TAU + Critic_target_state_dict[key] * (1 - self.TAU)
        self.policy_target.Critic.load_state_dict(Critic_target_state_dict)