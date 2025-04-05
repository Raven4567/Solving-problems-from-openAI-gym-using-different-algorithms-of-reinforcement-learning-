import numpy as np

class SARSA:
    def __init__(self, Action_dim: int, Observ_dim: int, episodes: int, eps_start: float = 0.995, eps_end: float = 0.005, gamma: float = 0.99, lr: float = 0.001):
        self.Q_table = np.random.uniform(low=0, high=0.05, size=(Observ_dim, Action_dim))

        self.lr = lr
        self.gamma = gamma

        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_end ** (1/episodes)

        self.action_dim = Action_dim
        self.observ_dim = Observ_dim
    
    def get_action(self, state: np.ndarray):
        if self.eps > np.random.rand():
            return np.random.randint(0, self.action_dim - 1)

        else:
            return self.Q_table[state].argmax()
    
    def update_Q_table(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray):
        self.Q_table[state, action] += self.lr * (reward + self.gamma * self.Q_table[next_state, self.get_action(next_state)] - self.Q_table[state, action])
    
    def call_epsilon_decay(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_end)