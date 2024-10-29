import torch
from torch import nn, tensor, optim, device, distributions
import torch.nn.functional as F
import numpy as np

device = device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, action_std_init: float, action_scaling: float, action_dim: int, observ_dim: int, has_continuous: bool):
        super().__init__()

        self.has_continuous = has_continuous # discrete or continuous
        
        if self.has_continuous:
            self.action_scaling = action_scaling # for scaling dist.sample() if you're using continuous PPO

            # action_std_init and action_var for exploration of environment
            self.action_std_init = action_std_init
            self.action_var = torch.full(size=(action_dim,), fill_value=action_std_init ** 2, device=device) 

            self.Actor = nn.Sequential(
                nn.Linear(observ_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, action_dim)
            ) # Initialization of actor if you're using continuous PPO

        else:
            self.Actor = nn.Sequential(
                nn.Linear(observ_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, action_dim),
                nn.Softmax(dim=-1)
            ) # Initialization of actor if you're using discrete PPO

        self.Critic = nn.Sequential(
            nn.Linear(observ_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        ) # Critic's initialization

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
        
        self.to(device)

    def forward(self, state: tensor):
        raise NotImplementedError

    def get_dist(self, state: tensor):
        if self.has_continuous:
            mu = self.Actor(state)
            cov_mat = torch.diag(self.action_var)

            dist = distributions.MultivariateNormal(mu, cov_mat)
        
        else:
            probs = self.Actor(state)
            dist = distributions.Categorical(probs)

        return dist
    
    def get_value(self, state: tensor):
        return self.Critic(state) 
    
    def get_evaluate(self, state: tensor, action: tensor):
        if self.has_continuous: # If continuous
            mu = self.Actor(state)

            action_var = self.action_var.expand_as(mu)
            cov_mat = torch.diag_embed(action_var).to(device)

            dist = distributions.MultivariateNormal(mu, cov_mat)
        
        else: # If discrete
            probs = self.Actor(state)
            dist = distributions.Categorical(probs)

        log_probs = dist.log_prob(action)
        value = self.Critic(state).squeeze(1) # using a .squeeze(1) to transform tensor with shape [200, 1] to vector with shape [200]
        dist_entropy = dist.entropy()

        return log_probs, value, dist_entropy

class PPO:
    def __init__(self, has_continuous: bool, Action_dim: int, Observ_dim: int, count_of_decay: int = None, action_scaling: float = None, Actor_lr: float = 0.0003, Critic_lr: float = 0.0025, action_std_init: float = None, action_std_min: float = None, gamma: float = 0.99, policy_clip: float = 0.2, k_epochs: int = 3, batch_size: int = None):
        # Initializing the most important attributes of PPO.
        self.policy = ActorCritic(action_std_init, action_scaling, Action_dim, Observ_dim, has_continuous)
        self.policy_old = ActorCritic(action_std_init, action_scaling, Action_dim, Observ_dim, has_continuous)

        self.Memory = Memory()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_fn = nn.SmoothL1Loss() # loss function, SmoothL1Loss for tasks of regression
        self.optimizer = optim.AdamW([ # Optimizer AdamW for Actor&Critic
            {'params': self.policy.Actor.parameters(), 'lr': Actor_lr, 'amsgrad': True},
            {'params': self.policy.Critic.parameters(), 'lr': Critic_lr, 'amsgrad': True}
        ])

        # Saving of collected hyperparameters, which u can get using PPO.your_hyperparameter,
        # it usefully, when you need get hyperparameters to Graphic class

        self.Actor_lr = Actor_lr
        self.Critic_lr = Critic_lr

        self.has_continuous = has_continuous

        if self.has_continuous:
            self.action_scaling = action_scaling

            self.count_of_decay = count_of_decay

            self.action_std = action_std_init
            self.action_std_min = action_std_min
            self.action_std_decay = action_std_min ** (1/count_of_decay)
        
        self.batch_size = batch_size

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.k_epochs = k_epochs

        self.action_dim = Action_dim
        self.observ_dim = Observ_dim

        #torch.autograd.set_detect_anomaly(mode=True, check_nan=True)

    def get_action(self, state: tensor):
        state = tensor(state, dtype=torch.float32, device=device) # Transform numpy state to tensor state

        with torch.no_grad(): # torch.no_grad() for economy of resource
            dist = self.policy_old.get_dist(state)

        # action = dist.sample and scaling if has_continuous, else just dist.smaple()
        action = F.tanh(dist.sample()) * self.action_scaling if self.has_continuous else dist.sample()
        value = self.policy_old.get_value(state).item()
        log_probs = dist.log_prob(action).item()

        action = [a.item() for a in action] if self.has_continuous else action.item()

        return action, value, log_probs

    def compute_gae(self, rewards, dones, values, next_value, GAE_lambda=0.95):
        # Just computing of GAE.

        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * GAE_lambda * (1 - dones[step]) * gae
            
            returns.insert(0, gae + values[step])

            next_value = values[step]

        return returns

    def education(self):
        old_states = tensor(np.array(self.Memory.states), dtype=torch.float32, device=device).detach()
        old_actions = tensor(np.array(self.Memory.actions), dtype=torch.float32, device=device).detach()
        old_log_probs = tensor(np.array(self.Memory.log_probs), dtype=torch.float32, device=device).detach()
        old_values = tensor(np.array(self.Memory.values), dtype=torch.float32, device=device).detach()
        
        rewards = self.Memory.rewards
        dones = self.Memory.dones

        # Computing GAE
        state_values = self.policy.get_value(old_states).squeeze(1)
        next_value = state_values[-1].detach()
        returns = self.compute_gae(rewards, dones, state_values, next_value)
        returns = tensor(returns, dtype=torch.float32, device=device).detach()
        
        # Normalazed rewards, or just GAE if we have only one element, 'cause it will lead to error
        returns = (returns - returns.mean()) / (returns.std() + 1e-7) if len(returns) > 1 else returns

        advantages = returns - old_values # calculate advantage
        for _ in range(self.k_epochs):
            # Collecting log probs, values of states, and dist entropy
            log_probs, state_values, dist_entropy = self.policy.get_evaluate(old_states, old_actions)
            
            # calculating and clipping of ratios, 'cause using of exp() function can will lead to inf or nan values
            ratios = torch.exp(torch.clamp(log_probs - old_log_probs, min=-20, max=20))

            surr1 = ratios * advantages # calculating of surr1
            
            # clipping of ratios, where min is 1 - policy_clip, and max is 1 + policy_clip, next multiplying on advantages
            surr2 = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * advantages 
            
            # gradient is loss of actor + 0.5 * loss of critic - 0.02 * dist_entropy. 0.02 is entropy bonus
            loss = -torch.min(surr1, surr2) + 0.5 * self.loss_fn(state_values, returns) - 0.02 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward() # using mean of loss to back propagation
            nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), 2) # cliping of actor parameters
            nn.utils.clip_grad_norm_(self.policy.Critic.parameters(), 2) # cliping of critic parameters
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict()) # load parameters of policy to policy_old

        self.Memory.clear()

    def call_action_std_decay(self):
        # Use it roughly after each episode, to transition from exploration to exploitation.
        
        if self.has_continuous: # if u using continuous PPO, all ok
            self.action_std = max(self.action_std * self.action_std_decay, self.action_std_min)
            
            new_action_var = torch.full(size=(self.action_dim,), fill_value=self.action_std ** 2, device=device) # update of action_var
        
            self.policy.action_var = new_action_var # updating at policy
            self.policy_old.action_var = new_action_var # updating at old policy
        
        else: # If you use discrete PPO and call PPO.call_action_std_decay, you will get an error.
            raise AttributeError('You can`t use PPO.call_action_std_decay with discrete PPO.')