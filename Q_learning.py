import numpy as np

class Q_learning:
    def __init__(self, action_dim: int, observ_dim: int, episodes: int, eps_start: float = 0.995, eps_end: float = 0.005, gamma: float = 0.995, lr: float = 0.001):
        self.Q_table = np.random.uniform(low=0, high=0.05, size=(observ_dim, action_dim))

        self.lr = lr
        self.gamma = gamma

        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_end ** (1/episodes)

        self.action_dim = action_dim
        self.observ_dim = observ_dim
    
    def get_action(self, state):
        if self.eps > np.random.rand():
            return np.random.randint(0, self.action_dim - 1)

        else:
            return self.Q_table[state].argmax()
    
    def update_Q_table(self, state, action, reward, next_state):
        self.Q_table[state, action] += self.lr * (reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state, action])
    
    def call_epsilon_decay(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_end)