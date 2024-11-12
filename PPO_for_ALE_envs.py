import torch as t
from torch import nn, tensor, optim, device, distributions
import torch.nn.functional as F

import numpy as np

device = device('cuda' if t.cuda.is_available() else 'cpu')

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]
        del self.log_probs[:]

class ActorCritic(nn.Module):
    def __init__(self, action_scaling: float, action_dim: int, observ_dim: int, has_continuous: bool):
        super().__init__()

        self.has_continuous = has_continuous # discrete or continuous
        
        if self.has_continuous:
            self.action_scaling = tensor(action_scaling, dtype=t.float64, device=device) # for scaling dist.sample() if you're using continuous PPO

            # action_std_init and action_var for exploration of environment
            #self.action_std_init = action_std_init
            #self.action_var = torch.full(size=(action_dim,), fill_value=action_std_init ** 2, device=device) 

            self.max_log_of_std = t.log(self.action_scaling)

            self.Actor = nn.Sequential(
                nn.Conv2d(observ_dim, 64, kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 64, kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 32, kernel_size=3, stride=2),

                nn.Flatten(),

                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(896, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)
                ),

                nn.Linear(128, 64),
                nn.ReLU()
                
            ) # Initialization of actor if you're using continuous PPO

            self.mu_layer = nn.Linear(64, action_dim) # mu_layer for getting mean of actions
            self.log_std = nn.Linear(64, action_dim) # log_std for gettinf log of standard deviation which we predicting

        else:
            self.Actor = nn.Sequential(
                nn.Conv2d(observ_dim, 64, kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 64, kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 32, kernel_size=3, stride=2),

                nn.Flatten(),

                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(896, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)
                ),

                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            ) # Initialization of actor if you're using discrete PPO

        self.Critic = nn.Sequential(
            nn.Conv2d(observ_dim, 64, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),

            nn.Flatten(),

            nn.Sequential(
                nn.ReLU(),
                nn.Linear(896, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            ),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ) # Critic's initialization

        if self.has_continuous: # If our sequential model split up on: Actor, mu_layer and log_std
            nn.init.xavier_uniform_(self.mu_layer.weight)
            nn.init.constant_(self.mu_layer.bias, 0)

            nn.init.xavier_uniform_(self.log_std.weight)
            nn.init.constant_(self.log_std.bias, 0)

        for layer in self.Actor: # xavier_initialization for actor 
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        for layer in self.Critic: # xavier_initialization for critic
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # If out sequential model is split up, we unite our parameters of actor, mu_layer, log_std else we just getting Actor.parameters()
        self.Actor_parameters = list(self.Actor.parameters()) + \
                                list(self.mu_layer.parameters()) + \
                                list(self.log_std.parameters()) \
                                if has_continuous else list(self.Actor.parameters())
            
        self.Critic_parameters = list(self.Critic.parameters()) # Critic_parameters for discrete or continuous PPO
        
        self.to(device) # Send of model to GPU or CPU

    def forward(self, state: tensor):
        raise NotImplementedError

    def get_dist(self, state: tensor):
        if self.has_continuous:
            features = self.Actor(state)

            mu = F.tanh(self.mu_layer(features)) * self.action_scaling
            std = F.softplus(t.clamp(self.log_std(features), min=-self.max_log_of_std, max=self.max_log_of_std))

            dist = distributions.Normal(mu, std)
        
        else:
            probs = self.Actor(state)
            dist = distributions.Categorical(probs)

        return dist
    
    def get_value(self, state: tensor):
        return self.Critic(state) 
    
    def get_evaluate(self, state: tensor, action: tensor):
        if self.has_continuous: # If continuous
            features = self.Actor(state)

            mu = F.tanh(self.mu_layer(features)) * self.action_scaling
            std = F.softplus(t.clamp(self.log_std(features), min=-self.max_log_of_std, max=self.max_log_of_std))

            dist = distributions.Normal(mu, std)
        
        else: # If discrete
            probs = self.Actor(state)
            dist = distributions.Categorical(probs)

        log_probs = dist.log_prob(action).sum(dim=-1) if self.has_continuous else dist.log_prob(action)
        value = self.get_value(state).squeeze(1) # using a .squeeze(1) to transform tensor with shape [x, 1] to vector with shape [200]
        dist_entropy = dist.entropy().sum(dim=-1) if self.has_continuous else dist.entropy()

        return log_probs, value, dist_entropy

class PPO:
    def __init__(self, has_continuous: bool, Action_dim: int, Observ_dim: int,  
                 action_scaling: float = None, Actor_lr: float = 0.001, Critic_lr: float = 0.0025, 
                 #count_of_decay: int = None, action_std_init: float = None, action_std_min: float = None, 
                 GAE_lambda: float = 0.95, gamma: float = 0.99, policy_clip: float = 0.2, k_epochs: int = 3, batch_size: int = None,
                 is_debugging: bool = False
                 ):

        # Initializing the most important attributes of PPO.
        self.policy = ActorCritic(action_scaling, Action_dim, Observ_dim, has_continuous)
        self.policy_old = ActorCritic(action_scaling, Action_dim, Observ_dim, has_continuous)

        self.Memory = Memory()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_fn = nn.MSELoss() # loss function, SmoothL1Loss for tasks of regression
        self.optimizer = optim.Adam([ # Optimizer AdamW for Actor&Critic
            {'params': self.policy.Actor_parameters, 'lr': Actor_lr},
            {'params': self.policy.Critic_parameters, 'lr': Critic_lr}
        ])

        # Saving of collected hyperparameters, which u can get using PPO.your_hyperparameter,
        # it usefully, when you need get hyperparameters to Graphic class

        self.Actor_lr = Actor_lr
        self.Critic_lr = Critic_lr

        self.has_continuous = has_continuous

        self.action_scaling = action_scaling

            #self.count_of_decay = count_of_decay 
            #
            #self.action_std = action_std_init
            #self.action_std_min = action_std_min
            #self.action_std_decay = action_std_min ** (1/count_of_decay)
        
        self.batch_size = batch_size

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.k_epochs = k_epochs
        self.GAE_lambda = GAE_lambda

        self.action_dim = Action_dim
        self.observ_dim = Observ_dim
        
        if is_debugging:
            t.autograd.set_detect_anomaly(mode=True, check_nan=True)

    def get_action(self, state: tensor):
        state = tensor(state, dtype=t.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0 # Transform numpy state to tensor state

        with t.no_grad(): # torch.no_grad() for economy of resource
            dist = self.policy_old.get_dist(state)

        # action = dist.sample and scaling if has_continuous, else just dist.smaple()
        action = F.tanh(dist.sample()) * self.action_scaling if self.has_continuous else dist.sample()
        value = self.policy_old.get_value(state).squeeze(0).item()
        log_prob = dist.log_prob(action).sum().item() if self.has_continuous else dist.log_prob(action).item()

        action = [a.item() for a in action] if self.has_continuous else action.item()

        return action, value, log_prob

    def compute_gae(self, rewards, dones, values, next_value):
        # Just computing of GAE.

        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.GAE_lambda * (1 - dones[step]) * gae
            
            returns.insert(0, gae + values[step])

            next_value = values[step]

        return returns

    def education(self):
        old_states = (tensor(np.array(self.Memory.states), dtype=t.float32, device=device).unsqueeze(1) / 255.0).detach()
        old_actions = tensor(np.array(self.Memory.actions), dtype=t.float32, device=device).detach()
        old_log_probs = tensor(np.array(self.Memory.log_probs), dtype=t.float32, device=device).detach()
        old_values = tensor(np.array(self.Memory.values), dtype=t.float32, device=device).detach()
        
        rewards = np.array(self.Memory.rewards)
        dones = np.array(self.Memory.dones)

        # Computing GAE
        state_values = self.policy.get_value(old_states).squeeze(dim=-1)
        next_value = state_values[-1].detach()
        returns = self.compute_gae(rewards, dones, state_values, next_value)
        returns = tensor(returns, dtype=t.float32, device=device).detach()
        
        # Normalazed rewards, or just GAE if we have only one element, 'cause it will lead to error
        # returns = (returns - returns.mean()) / (returns.std() + 1e-7) if len(returns) > 1 else returns

        advantages = returns - old_values # calculate advantage
        for _ in range(self.k_epochs):

            # Collecting log probs, values of states, and dist entropy
            log_probs, state_values, dist_entropy = self.policy.get_evaluate(old_states, old_actions)
            
            # calculating and clipping of log_probs, 'cause using of exp() function can will lead to inf or nan values
            ratios = t.exp(t.clamp(log_probs - old_log_probs, min=-20, max=20))

            surr1 = ratios * advantages # calculating of surr1
            surr2 = t.clamp(ratios, min=1 - self.policy_clip, max=1 + self.policy_clip) * advantages  # clipping of ratios, where min is 1 - policy_clip, and max is 1 + policy_clip, next multiplying on advantages
            
            # gradient is loss of actor + 0.5 * loss of critic - 0.02 * dist_entropy. 0.02 is entropy bonus
            loss = -t.min(surr1, surr2) + 0.5 * self.loss_fn(state_values, returns) - 0.02 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward() # using mean of loss to back propagation
            nn.utils.clip_grad_value_(self.policy.Actor_parameters, 100) # cliping of actor parameters
            nn.utils.clip_grad_value_(self.policy.Critic_parameters, 100) # cliping of critic parameters
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict()) # load parameters of policy to policy_old

        self.Memory.clear()