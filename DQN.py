import torch as t
from torch import nn, optim

import numpy as np

from random import sample, random
from collections import deque

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.memory = deque([], maxlen=maxlen)
    
    def push(self, data: list):
        self.memory.append([np.array(x) for x in data])
    
    def sample_batch(self, batch_size: int):
        return sample(self.memory, k=batch_size)
    
    def __len__(self):
        return len(self.memory)

class Network(nn.Module):
    def __init__(self, action_dim: int, observ_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(observ_dim, 64),
            nn.ReLU6(inplace=True),

            nn.Linear(64, 64),
            nn.ReLU6(inplace=True),

            nn.Linear(64, action_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0, std=0.01)
                    
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.to(device)
    
    def forward(self, state: t.Tensor):
        return self.model(state)

class DQN:
    def __init__(self, action_dim: int, observ_dim: int, count_of_decay: int, lr: float = 0.001, eps_start: float = 0.995, eps_end: float = 0.005, maxlen_of_buffer: int = 50000, batch_size: int = 64, gamma: float = 0.995, TAU: float = 0.005):
        self.policy_net = Network(action_dim, observ_dim)
        self.target_net = Network(action_dim, observ_dim)
        self.buffer = ReplayBuffer(maxlen_of_buffer)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = self.eps_end ** (1/count_of_decay)

        self.batch_size = batch_size
        self.gamma = gamma
        self.TAU = TAU

        self.lr = lr

        self.action_dim = action_dim
        self.observ_dim = observ_dim
    
    @t.no_grad()
    def get_action(self, state: t.Tensor):
        if self.eps > random():
            action = t.randint(0, self.action_dim).to(device)

        else:
            state = state.to(dtype=t.float32, device=device)
            
            action = self.policy_net(state).cpu().argmax().numpy()
            
        return action
    
    def call_epsilon_decay(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_min)
    
    def education(self):
        if len(self.buffer) < self.batch_size:
            return 
        
        batch = self.buffer.sample_batch(self.batch_size)

        states, actions, rewards, next_states = zip(*batch)

        states = t.from_numpy(np.array(states)).to(dtype=t.float32, device=device)
        actions = t.from_numpy(np.array(actions)).to(dtype=t.int64, device=device)
        rewards = t.from_numpy(np.array(rewards)).to(dtype=t.float32, device=device)
        next_states = t.from_numpy(np.array(next_states)).to(dtype=t.float32, device=device)

        with t.no_grad():
            target_Q_values = self.target_net(next_states).max(dim=1)[0]

            target_Q_values = rewards + self.gamma * target_Q_values
        Q_values = self.policy_net(states).gather(1, actions)

        loss = self.loss_fn(Q_values, target_Q_values.unsqueeze(1))

        self.optimizer.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)

        self.optimizer.step()

    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)