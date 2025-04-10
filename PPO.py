import torch as t
from torch import nn, optim, distributions
from torch.nn import functional as F

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
    
    def push(self, state, action, reward, done, state_value, log_prob):
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.rewards.append(np.array(reward, dtype=np.float32))
        self.dones.append(np.array(done, dtype=np.float32))
        self.state_values.append(np.array(state_value, dtype=np.float32))
        self.log_probs.append(np.array(log_prob, dtype=np.float32))
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.state_values[:]
        del self.log_probs[:]

class ActorCritic(nn.Module):
    def __init__(self, action_dim: int, observ_dim: int, has_continuous: bool, action_scaling: float):
        super().__init__()

        self.has_continuous = has_continuous # discrete or continuous

        if self.has_continuous:
            self.action_scaling = t.tensor(action_scaling, dtype=t.float32, device=device) # for scaling dist.sample() if you're using continuous PPO

            self.log_std = t.log(self.action_scaling)
                
            self.Actor = nn.Sequential(
                nn.Linear(observ_dim, 64),
                nn.GroupNorm(64 // 8, 64),
                nn.ReLU6(inplace=True),

                nn.Linear(64, 64),
                nn.GroupNorm(64 // 8, 64),
                nn.ReLU6(inplace=True),
            ) # Initialization of actor if you're using continuous PPO

            self.mu_layer = nn.Linear(64, action_dim) # mu_layer for getting mean of actions
            self.log_std_layer = nn.Linear(64, action_dim) # log_std for gettinf log of standard deviation which we predicting

        else:
            self.Actor = nn.Sequential(
                nn.Linear(observ_dim, 64),
                nn.GroupNorm(64 // 8, 64),
                nn.ReLU6(inplace=True),

                nn.Linear(64, 64),
                nn.GroupNorm(64 // 8, 64),
                nn.ReLU6(inplace=True),

                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            ) # Initialization of actor if you're using discrete PPO

        self.Critic = nn.Sequential(
            nn.Linear(observ_dim, 64),
            nn.GroupNorm(64 // 8, 64),
            nn.ReLU6(inplace=True),

            nn.Linear(64, 64),
            nn.GroupNorm(64 // 8, 64),
            nn.ReLU6(inplace=True),

            nn.Linear(64, 1)
        ) # Critic's initialization

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0, std=0.01)
                    
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
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
            std = F.softplus(t.clamp(self.log_std_layer(features), min=-self.log_std, max=self.log_std))

            dist = distributions.Normal(mu, std)
        
        else:
            probs = self.Actor(state)
            dist = distributions.Categorical(probs)

        return dist
    
    def get_value(self, state: t.Tensor):
        return self.Critic(state).squeeze(-1)
    
    def get_evaluate(self, states: t.Tensor, actions: t.Tensor):
        dist = self.get_dist(states)

        log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        if self.has_continuous:
            log_probs = log_probs.sum(-1)
            dist_entropy = dist_entropy.sum(-1)
        else:
            pass
        
        state_value = self.get_value(states)

        return log_probs, state_value, dist_entropy

class RND(nn.Module):
    def __init__(self, in_features: int, out_features: int, beta: int = 0.01, k_epochs: int = 3):
        super().__init__()

        self.target_net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GroupNorm(64 // 8, 64),
            nn.ReLU6(),

            nn.Linear(64, out_features)
        )

        self.pred_net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GroupNorm(64 // 8, 64),
            nn.ReLU6(),

            nn.Linear(64, out_features)
        )
        
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.beta = t.tensor([beta], dtype=t.float32, device=device)
        self.k_epochs = k_epochs

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.pred_net.parameters(), lr=0.001)
        
        self.to(device)
    
    def compute_intristic_reward(self, values: list):
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

        intristic_rewards = t.norm(pred_batches - target_batches, dim=-1)

        self.update_pred(values)

        return intristic_rewards * self.beta
    
    def update_pred(self, values: list):
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
            self, has_continuous: bool, action_dim: int, observ_dim: int,  
            Actor_lr: float = 0.001, Critic_lr: float = 0.0025, action_scaling: float = None, 
            k_epochs: int = 21, policy_clip: float = 0.2, GAE_lambda: float = 0.95,
            gamma: float = 0.995, batch_size: int = 1024, mini_batch_size: int = 512, 
            use_RND: bool = False, beta: int = None
        ):

        # Initializing the most important attributes of PPO.
        self.policy = ActorCritic(action_dim, observ_dim, has_continuous, action_scaling)
        self.policy_old = ActorCritic(action_dim, observ_dim, has_continuous, action_scaling)
        if use_RND:
            self.rnd = RND(in_features=observ_dim, out_features=observ_dim, beta=beta)

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

        self.policy_clip = policy_clip
        self.k_epochs = k_epochs
        self.GAE_lambda = GAE_lambda

        self.use_RND = use_RND
        self.beta = beta

        self.action_dim = action_dim
        self.observ_dim = observ_dim

    @t.no_grad()
    def get_action(self, state: t.Tensor):
        state = state.to(dtype=t.float32, device=device) # Transform numpy state to tensor state
        
        dist = self.policy_old.get_dist(state)

        action = dist.sample()
        if self.has_continuous:
            action = t.tanh(action) * self.action_scaling
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

    def learn(self):
        if len(self.memory.states) < self.batch_size:
            return 

        old_states = t.from_numpy(np.array(self.memory.states)).to(device).detach()
        old_actions = t.from_numpy(np.array(self.memory.actions)).to(device).detach()
        old_log_probs = t.from_numpy(np.array(self.memory.log_probs)).to(device).detach()
        old_state_values = t.from_numpy(np.array(self.memory.state_values)).to(device).detach()
        
        if self.use_RND:
            rewards = (
                t.from_numpy(np.array(self.memory.rewards)).to(device) + \
                self.rnd.compute_intristic_reward(
                    self.batch_packer(old_states, self.mini_batch_size)
                    )
                ).cpu().numpy()

        else:
            rewards = np.array(self.memory.rewards)
        dones = np.array(self.memory.dones)

        self.memory.clear()

        # Computing GAE
        state_values = old_state_values.cpu().numpy()
        next_value = state_values[-1]
        
        returns = self.compute_gae(rewards, dones, state_values, next_value)
        returns = t.from_numpy(np.array(returns)).to(dtype=t.float32, device=device).detach()

        advantages = returns - old_state_values # calculate advantage

        batches = self.batch_packer([old_states, old_actions, old_log_probs, advantages, returns], batch_size=self.batch_size)
        
        for _ in range(self.k_epochs):
            for batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns in zip(*batches):

                mini_batches = self.batch_packer([batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns], batch_size=self.mini_batch_size)
                
                for mini_batch_states, mini_batch_actions, mini_batch_log_probs, mini_batch_advantages, mini_batch_returns in zip(*mini_batches):
                    # Collecting log probs, values of states, and dist entropy
                    log_probs, state_values, dist_entropy = self.policy.get_evaluate(mini_batch_states, mini_batch_actions)
                            
                    # calculating and clipping of log_probs, 'cause using of exp() function can will lead to inf or nan values
                    ratios = t.exp(t.clamp(log_probs - mini_batch_log_probs, min=-20, max=20))

                    surr1 = ratios * mini_batch_advantages # calculating of surr1
                    surr2 = t.clamp(ratios, min=1 - self.policy_clip, max=1 + self.policy_clip) * mini_batch_advantages  # clipping of ratios, where min is 1 - policy_clip, and max is 1 + policy_clip, next multiplying on advantages
                            
                    # gradient is loss of actor + 0.5 * loss of critic - 0.02 * dist_entropy. 0.02 is entropy bonus
                    loss = -t.min(surr1, surr2) + 0.5 * self.loss_fn(state_values, mini_batch_returns) - 0.02 * dist_entropy
                    
                    self.optimizer.zero_grad()

                    loss.mean().backward() # using mean of loss to back propagation

                    nn.utils.clip_grad_value_(self.policy.Actor_parameters, 100) # cliping of actor parameters
                    nn.utils.clip_grad_value_(self.policy.Critic_parameters, 100) # cliping of critic parameters           
                    
                    self.optimizer.step()

        t.cuda.empty_cache() if device == 'cuda' else None

        self.policy_old.load_state_dict(self.policy.state_dict()) # load parameters of policy to policy_old

    def load_weights(self):
        try:
            self.policy.load_state_dict(t.load('./Policy_weights.pth', weights_only=True))
            self.policy_old.load_state_dict(self.policy.state_dict())
            
            if self.use_RND:            
                self.rnd.load_state_dict(t.load('./RND_weights.pth', weights_only=True))

        except FileNotFoundError:
            pass
    
    def save_weights(self):
        t.save(self.policy.state_dict(), './Policy_weights.pth')

        if self.use_RND:
            t.save(self.rnd.state_dict(), './RND_weights.pth')