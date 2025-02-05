import torch as t
from torch import nn, tensor, optim, distributions
import torch.nn.functional as F

from torchvision import models

import numpy as np

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.state_values = []
        self.log_probs = []

    def push(self, state, action, reward, done, value, log_prob):
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.rewards.append(np.array(reward, dtype=np.float32))
        self.dones.append(np.array(done, dtype=np.float32))
        self.state_values.append(np.array(value, dtype=np.float32))
        self.log_probs.append(np.array(log_prob, dtype=np.float32))

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.state_values[:]
        del self.log_probs[:]

class ActorCritic(nn.Module):
    def __init__(self, action_scaling: float, action_dim: int, in_channels: int, has_continuous: bool):
        super().__init__()

        self.has_continuous = has_continuous # discrete or continuous
        
        if self.has_continuous:
            self.action_scaling = tensor(action_scaling, dtype=t.float32, device=device) # for scaling dist.sample() if you're using continuous PPO

            # action_std_init and action_var for exploration of environment
            #self.action_std_init = action_std_init
            #self.action_var = torch.full(size=(action_dim,), fill_value=action_std_init ** 2, device=device) 

            self.max_log_of_std = t.log(self.action_scaling)

            self.Actor = models.mobilenet_v3_small() # Initialization of actor if you're using continuous PPO
            self.Actor.features[0][0] = nn.Conv2d(in_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.Actor.classifier = self.Actor.classifier[:-1]

            self.mu_layer = nn.Linear(1024, action_dim) # mu_layer for getting mean of actions
            self.log_std = nn.Linear(1024, action_dim) # log_std for gettinf log of standard deviation which we predicting

        else:
            self.Actor = models.mobilenet_v3_small() # Initialization of actor if you're using discrete PPO
            self.Actor.features[0][0] = nn.Conv2d(in_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.Actor.classifier[-1] = nn.Linear(1024, action_dim)
            self.Actor.classifier = nn.Sequential(*list(self.Actor.classifier), nn.Softmax(dim=-1))

        self.Critic = models.mobilenet_v3_small()
        self.Critic.features[0][0] = nn.Conv2d(in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.Critic.classifier[-1] = nn.Linear(1024, 1)

        if self.has_continuous: # If our sequential model split up on: Actor, mu_layer and log_std
            nn.init.xavier_uniform_(self.mu_layer.weight)
            nn.init.constant_(self.mu_layer.bias, 0)

            nn.init.xavier_uniform_(self.log_std.weight)
            nn.init.constant_(self.log_std.bias, 0)

        for layer in self.Actor.features: # xavier_initialization for actor 
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        for layer in self.Actor.classifier: # xavier_initialization for actor 
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        for layer in self.Critic.features: # xavier_initialization for critic
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        for layer in self.Critic.classifier: # xavier_initialization for critic
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

class RND(nn.Module):
    def __init__(self, in_features: int, out_features: int, beta: int = 0.02, k_epochs: int = 3):
        super().__init__()

        self.target_net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU6(),
            nn.Linear(64, out_features)
        )

        self.pred_net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU6(),
            nn.Linear(64, out_features)
        )
        
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.beta = tensor([beta], dtype=t.float32, device=device)
        self.k_epochs = k_epochs

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.pred_net.parameters(), lr=0.001)
        
        self.to(device)
    
    def compute_intristic_reward(self, values):
        target_batches = []
        pred_batches = []

        for i in values:
            with t.no_grad():
                targets = self.target_net(i)
                preds = self.pred_net(i)

            target_batches.append(targets)
            pred_batches.append(preds)
        
        target_batches = t.cat(target_batches, dim=0)
        pred_batches = t.cat(pred_batches, dim=0)

        reward = t.norm(pred_batches - target_batches, dim=-1)

        self.update_pred(values)

        return reward * self.beta
    
    def update_pred(self, values):
        self.pred_net.train()
        
        for _ in range(self.k_epochs):
            for i in values:
                with t.no_grad():
                    targets = self.target_net(i)
                preds = self.pred_net(i)

                loss = self.loss_fn(preds, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.pred_net.eval()

class PPO:
    def __init__(self, has_continuous: bool, Action_dim: int, in_channels: int,  
                 action_scaling: float = None, Actor_lr: float = 0.001, Critic_lr: float = 0.0025, 
                 k_epochs: int = 23, policy_clip: float = 0.2, GAE_lambda: float = 0.95,
                 gamma: float = 0.995, batch_size: int = 1024, mini_batch_size: int = 512, 
                 use_RND: bool = False, beta: int = None
                 ):

        # Initializing the most important attributes of PPO.
        self.policy = ActorCritic(action_scaling, Action_dim, in_channels, has_continuous)
        self.policy_old = ActorCritic(action_scaling, Action_dim, in_channels, has_continuous)
        if use_RND:
            self.rnd = RND(in_channels, out_features=32, beta=beta)
        
        self.policy = t.compile(self.policy)
        self.policy_old = t.compile(self.policy_old)
        if use_RND:
            self.rnd = t.compile(self.rnd)

        self.memory = Memory()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.policy.train()
        self.policy_old.eval()
        if use_RND:
            self.rnd.eval()

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
        
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma

        self.use_RND = use_RND
        self.beta = beta

        self.policy_clip = policy_clip
        self.k_epochs = k_epochs
        self.GAE_lambda = GAE_lambda

        self.action_dim = Action_dim
        self.in_channels = in_channels

    def get_action(self, state: tensor):
        state = tensor(state, dtype=t.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0 # Transform numpy state to tensor state

        with t.no_grad(): # torch.no_grad() for economy of resource
            dist = self.policy_old.get_dist(state)

            # action = dist.sample and scaling if has_continuous, else just dist.smaple()
            action = F.tanh(dist.sample()) * self.action_scaling if self.has_continuous else dist.sample()
            state_value = self.policy_old.get_value(state).squeeze(0)
            log_prob = dist.log_prob(action).sum().item() if self.has_continuous else dist.log_prob(action)

        return action.cpu().numpy(), state_value.cpu().numpy(), log_prob.cpu().numpy()

    def batch_packer(self, values: list, batch_size: int):
        batch = []

        values = [i.detach().cpu().numpy() for i in values]

        mini_batches = [[] for _ in range(len(values))]

        while len(values[0]) > 0:
            unique_values_indexes = np.random.choice(a=np.arange(len(values[0])), replace=False, size=np.minimum(len(values[0]), batch_size))

            elements_for_mini_batches = [value[unique_values_indexes] for value in values]

            values = [np.delete(value, unique_values_indexes, axis=0) for value in values]

            [mini_batches[index].append(t.from_numpy(elements_for_mini_batches[index]).to(device)) for index in range(len(values))]
        
        [batch.append(mini_batches[index]) for index in range(len(values))]

        return batch

    def single_batch_packer(self, value: tensor, batch_size: int):
        value = value.detach().cpu().numpy()

        mini_batches = []

        while len(value) > 0:
            unique_values_indexes = np.random.choice(a=np.arange(len(value)), replace=False, size=np.minimum(len(value), batch_size))

            element_for_mini_batch = value[unique_values_indexes]
    
            value = np.delete(value, unique_values_indexes, axis=0)

            mini_batches.append(t.from_numpy(element_for_mini_batch).to(device))

        return mini_batches

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
        if len(self.memory.states) < self.batch_size:
            return 

        # Copy data
        old_states = t.from_numpy(np.array(self.memory.states)).to(device).unsqueeze(1).detach()
        old_actions = t.from_numpy(np.array(self.memory.actions)).to(device).detach()
        old_log_probs = t.from_numpy(np.array(self.memory.log_probs)).to(device).detach()
        old_values = t.from_numpy(np.array(self.memory.state_values)).to(device).detach()
        
        if self.use_RND:
            rewards = (
                t.from_numpy(np.array(self.memory.rewards)).to(device) + \
                self.rnd.compute_intristic_reward(
                    self.single_batch_packer(old_states, self.mini_batch_size)
                    )
                ).detach().cpu().numpy()
        else:
            rewards = np.array(self.memory.rewards)
        dones = np.array(self.memory.dones)

        # Clear copied data
        self.memory.clear()

        # Computing GAE
        state_values = t.cat([self.policy.get_value(i).squeeze(dim=-1) for i in self.single_batch_packer(old_states, self.mini_batch_size)], dim=0).detach().cpu().numpy()
        next_value = state_values[-1]
        returns = self.compute_gae(rewards, dones, state_values, next_value)
        returns = tensor(returns, dtype=t.float32, device=device).detach()

        advantages = returns - old_values # calculate advantage

        batch = self.batch_packer([old_states, old_actions, old_log_probs, advantages, returns], batch_size=self.batch_size)
        
        # K_epochs cycle
        for _ in range(self.k_epochs):
            
            t.cuda.empty_cache() if device == 'cuda' else 0

            for old_states_for_batches, old_actions_for_batches, old_log_probs_for_batches, advantages_for_batches, returns_for_batches in zip(*batch):    
                self.optimizer.zero_grad()

                mini_batch = self.batch_packer([old_states_for_batches, old_actions_for_batches, old_log_probs_for_batches, returns_for_batches, advantages_for_batches], batch_size=self.mini_batch_size)

                for batch_states, batch_actions, batch_log_probs, batch_returns, batch_advantages in zip(*mini_batch):
                    # Collecting log probs, values of states, and dist entropy
                    log_probs, state_values, dist_entropy = self.policy.get_evaluate(batch_states, batch_actions)
                            
                    # calculating and clipping of log_probs, 'cause using of exp() function can will lead to inf or nan values
                    ratios = t.exp(t.clamp(log_probs - batch_log_probs, min=-20, max=20))

                    surr1 = ratios * batch_advantages # calculating of surr1
                    surr2 = t.clamp(ratios, min=1 - self.policy_clip, max=1 + self.policy_clip) * batch_advantages  # clipping of ratios, where min is 1 - policy_clip, and max is 1 + policy_clip, next multiplying on advantages
                            
                    # gradient is loss of actor + 0.5 * loss of critic - 0.02 * dist_entropy. 0.02 is entropy bonus
                    loss = -t.min(surr1, surr2) + 0.5 * self.loss_fn(state_values, batch_returns) - 0.02 * dist_entropy

                    loss.mean().backward() # using mean of loss to back propagation
                
                nn.utils.clip_grad_value_(self.policy.Actor_parameters, 100) # cliping of actor parameters
                nn.utils.clip_grad_value_(self.policy.Critic_parameters, 100) # cliping of critic parameters
                    
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict()) # load parameters of policy to policy_old
    
    def load_weights(self, storage_path: str):
        try:
            self.policy.load_state_dict(t.load(storage_path+'Policy_weights.pth', weights_only=True))
            self.policy_old.load_state_dict(self.policy.state_dict())

            self.rnd.load_state_dict(t.load(storage_path+'RND_weights.pth', weights_only=True))
        except FileNotFoundError:
            pass
    
    def save_weights(self, storage_path: str):
        t.save(self.policy.state_dict(), storage_path+'Policy_weights.pth')
        t.save(self.rnd.state_dict(), storage_path+'RND_weights.pth')