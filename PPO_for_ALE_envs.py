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
    def __init__(self, action_dim: int, in_channels: int, has_continuous: bool, action_scaling: float):
        super().__init__()

        self.has_continuous = has_continuous # discrete or continuous
        
        if self.has_continuous:
            self.action_scaling = tensor(action_scaling, dtype=t.float32, device=device) # for scaling dist.sample() if you're using continuous PPO

            # action_std_init and action_var for exploration of environment
            #self.action_std_init = action_std_init
            #self.action_var = torch.full(size=(action_dim,), fill_value=action_std_init ** 2, device=device) 

            self.log_of_std = t.log(self.action_scaling)

            self.Actor = models.mobilenet_v3_small() # Initialization of actor if you're using continuous PPO
            self.Actor.features[0][0] = nn.Conv2d(in_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.Actor.classifier = self.Actor.classifier[:-1]

            self.mu_layer = nn.Linear(1024, action_dim) # mu_layer for getting mean of actions
            self.log_std_layer = nn.Linear(1024, action_dim) # log_std for gettinf log of standard deviation which we predicting

        else:
            self.Actor = models.mobilenet_v3_small() # Initialization of actor if you're using discrete PPO
            self.Actor.features[0][0] = nn.Conv2d(in_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.Actor.classifier[-1] = nn.Linear(1024, action_dim)
            self.Actor.classifier = nn.Sequential(*list(self.Actor.classifier), nn.Softmax(dim=-1))

        self.Critic = models.mobilenet_v3_small()
        self.Critic.features[0][0] = nn.Conv2d(in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.Critic.classifier[-1] = nn.Linear(1024, 1)

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
        
        # If out sequential model is split up, we unite our parameters of actor, mu_layer, log_std else we just getting Actor.parameters()
        self.Actor_parameters = list(self.Actor.parameters()) + \
                                list(self.mu_layer.parameters()) + \
                                list(self.log_std_layer.parameters()) \
                                if has_continuous else list(self.Actor.parameters())
            
        self.Critic_parameters = list(self.Critic.parameters()) # Critic_parameters for discrete or continuous PPO
        
        self.to(device) # Send of model to GPU or CPU

    def forward(self, state: t.Tensor):
        raise NotImplementedError

    def get_dist(self, state: t.Tensor):
        if self.has_continuous:
            features = self.Actor(state)

            mu = F.tanh(self.mu_layer(features)) * self.action_scaling
            std = F.softplus(t.clamp(self.log_std(features), min=-self.log_of_std, max=self.log_of_std))

            dist = distributions.Normal(mu, std)
        
        else:
            probs = self.Actor(state)
            dist = distributions.Categorical(probs)

        return dist
    
    def get_value(self, state: t.Tensor):
        return self.Critic(state).squeeze(-1)
    
    def get_evaluate(self, state: t.Tensor, action: t.Tensor):
        if self.has_continuous: # If continuous
            features = self.Actor(state)

            mu = F.tanh(self.mu_layer(features)) * self.action_scaling
            std = F.softplus(t.clamp(self.log_std(features), min=-self.log_of_std, max=self.log_of_std))

            dist = distributions.Normal(mu, std)

            log_probs = dist.log_prob(action).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
        
        else: # If discrete
            probs = self.Actor(state)
            dist = distributions.Categorical(probs)

            log_probs = dist.log_prob(action)
            dist_entropy = dist.entropy()

        state_value = self.get_value(state)

        return log_probs, state_value, dist_entropy

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
    def __init__(
            self, has_continuous: bool, action_dim: int, in_channels: int,  
            Actor_lr: float = 0.001, Critic_lr: float = 0.0025, action_scaling: float = None,
            k_epochs: int = 23, policy_clip: float = 0.2, GAE_lambda: float = 0.95,
            gamma: float = 0.995, batch_size: int = 1024, mini_batch_size: int = 512, 
            use_RND: bool = False, beta: int = None
        ):

        # Initializing the most important attributes of PPO.
        self.policy = ActorCritic(action_dim, in_channels, has_continuous, action_scaling)
        self.policy_old = ActorCritic(action_dim, in_channels, has_continuous, action_scaling)
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
        self.optimizer = optim.AdamW([ # Optimizer AdamW for Actor&Critic
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

        self.action_dim = action_dim
        self.in_channels = in_channels

    def get_action(self, state: t.Tensor):
        state = state.to(dtype=t.float32, device=device) # Transform numpy state to tensor state

        with t.no_grad(): # torch.no_grad() for economy of resource
            dist = self.policy_old.get_dist(state)

            action = dist.sample()
            if self.has_continuous:
                action = F.tanh(action) * self.action_scaling
                log_prob = dist.log_prob(action).sum(-1)
            else:
                log_prob = dist.log_prob(action)

            state_value = self.policy_old.get_value(state)

            return action.squeeze(0).cpu().numpy(), state_value.squeeze(0).cpu().numpy(), log_prob.squeeze(0).cpu().numpy()

    def batch_packer(self, values, batch_size: int):
        if isinstance(values, t.Tensor):
            batch = list(t.utils.data.DataLoader(values, batch_size))
        
        elif isinstance(values, list):
            batch = [list(t.utils.data.DataLoader(value, batch_size)) for value in values]

        return batch

    def compute_gae(self, rewards: np.ndarray, dones: np.ndarray, state_values: np.ndarray, next_value: np.ndarray):
        # Just computing of GAE.

        gae = 0
        returns = []
        for step in reversed(range(len(state_values))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - state_values[step]
            gae = delta + self.gamma * self.GAE_lambda * (1 - dones[step]) * gae
            
            returns.insert(0, gae + state_values[step])

            next_value = state_values[step]

        return returns

    def clip_memory(self):
        """This function is needed to prevent the situation 
        where batch_tensor.shape[0] = 1, because if
        we feed a batch with size 1 into a model with enabled the train() mode 
        we will get the error.
        
        The function takes all lists from self.memory and checks if the length of any list
        is a multiple of batch_size. If any list is bigger than self.batch_size by exactly 1 item, 
        then the last item is deleted."""
        
        values = self.memory.states, self.memory.actions, self.memory.log_probs, self.memory.rewards, self.memory.dones, self.memory.state_values

        new_values = []
        for single_list in values:
            # Check if the length of the list is a multiple of batch_size
            if (len(single_list)-1) % self.batch_size == 0:
                # If it is, remove the last element
                new_values.append(single_list[:-1])
            
            else:
                # If it is not, do nothing
                new_values.append(single_list)
        
        self.memory.states, self.memory.actions, self.memory.log_probs, self.memory.rewards, self.memory.dones, self.memory.state_values = new_values
    
    def education(self):
        if len(self.memory.states) < self.batch_size:
            return 

        self.clip_memory()

        # Copy data
        old_states = t.from_numpy(np.array(self.memory.states)).to(device).detach()
        old_actions = t.from_numpy(np.array(self.memory.actions)).to(device).detach()
        old_log_probs = t.from_numpy(np.array(self.memory.log_probs)).to(device).detach()
        old_values = t.from_numpy(np.array(self.memory.state_values)).to(device).detach()
        
        if self.use_RND:
            rewards = (
                t.from_numpy(np.array(self.memory.rewards)).to(device) + \
                self.rnd.compute_intristic_reward(
                    self.batch_packer(old_states, self.mini_batch_size)
                    )
                ).detach().cpu().numpy()
        else:
            rewards = np.array(self.memory.rewards)
        dones = np.array(self.memory.dones)

        # Clear copied data
        self.memory.clear()

        # Computing GAE
        state_values = t.cat([self.policy.get_value(i) for i in self.batch_packer(old_states, self.mini_batch_size)], dim=0).detach().cpu().numpy()
        next_value = state_values[-1]
        returns = self.compute_gae(rewards, dones, state_values, next_value)
        returns = t.from_numpy(np.array(returns)).to(dtype=t.float32, device=device).detach()

        advantages = returns - old_values # calculate advantage

        batches = self.batch_packer([old_states, old_actions, old_log_probs, advantages, returns], batch_size=self.batch_size)
        
        # K_epochs cycle
        for _ in range(self.k_epochs):
            for batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns in zip(*batches):

                mini_batches = self.batch_packer([batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns], batch_size=self.mini_batch_size)
                
                for mini_batch_states, mini_batch_actions, mini_batch_log_probs, mini_batch_advantages, mini_batch_returns in zip(*mini_batches):
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

        t.cuda.empty_cache() if device == 'cuda' else None

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