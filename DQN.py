import torch
from torch import nn, tensor, optim, device
import numpy as np
from random import sample, random, randint
from collections import deque

device = device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def push(self, data: list):
        self.memory.append([np.array(x) for x in data])
    
    def sample_batch(self, batch_size: int = 64):
        return sample(self.memory, k=batch_size)
    
    def __len__(self):
        return len(self.memory)

class Network(nn.Module):
    def __init__(self, action_dim: int, observ_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(observ_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        self.to(device)
    
    def forward(self, state: tensor):
        return self.model(state)

class DQN:
    def __init__(self, Action_dim: int, Observ_dim: int, episodes: int, lr: float = 0.001, eps: float = 0.995, eps_min: float = 0.005, maxlen_of_buffer: int = 50000, batch_size: int = 64, gamma: float = 0.99, TAU: float = 0.005):
        self.policy_net = Network(Action_dim, Observ_dim)
        self.target_net = Network(Action_dim, Observ_dim)
        self.buffer = ReplayBuffer(maxlen_of_buffer)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = self.eps_min ** (1/episodes)

        self.batch_size = batch_size
        self.gamma = gamma
        self.TAU = TAU

        self.lr = lr

        self.episodes = episodes

        self.action_dim = Action_dim
        self.observ_dim = Observ_dim
    
    def get_action(self, state):
        if self.eps > random():
            action = randint(0, self.action_dim - 1)

        else:
            state = tensor(state, dtype=torch.float32, device=device)

            with torch.no_grad():
                action = self.policy_net(state).argmax().item()
            
        return action
    
    def call_epsilon_decay(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_min)
    
    def education(self):
        if len(self.buffer) < self.batch_size:
            return 
        
        batch = self.buffer.sample_batch(self.batch_size)

        states, actions, rewards, next_states = zip(*batch)

        states = tensor(np.array(states), dtype=torch.float32, device=device).unsqueeze(1)
        actions = tensor(np.array(actions), dtype=torch.int64, device=device)
        rewards = tensor(np.array(rewards), dtype=torch.float32, device=device)
        next_states = tensor(np.array(next_states), dtype=torch.float32, device=device).unsqueeze(1)

        with torch.no_grad():
            target_Q_values = self.target_net(next_states).max(dim=1)[0]

            target_Q_values = rewards + self.gamma * target_Q_values
        Q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        loss = self.loss_fn(Q_values, target_Q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)
        self.optimizer.step()

    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)