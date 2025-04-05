# Solving-the-problem-from-openAI-gym-using-different-algorithms-of-reinforcement-learning-

This repository is a collection of various reinforcement learning algorithms, ranging from simple Q-learning to PPO (Proximal Policy Optimization), designed for training in OpenAI Gym environments.

For data visualization, the code uses matplotlib charts, with a moving average added for smoother visuals. The original data graph is semi-transparent, while the smoothed graph has a more vivid color.

# Usage:

## Q-learning:
```python
from Q_learning import Q_learning

q_learning = Q_learning(action_dim=env.action_space.n, observ_dim=env.observation_space.n, 
                        episodes=EPISODES, eps_start=0.995, eps_end=0.005, gamma=0.995, lr=0.001)
```

## SARSA:
```python
from SARSA import SARSA

sarsa = SARSA(action_dim=env.action_space.n, observ_dim=env.observation_space.n, episodes=EPISODES, 
              eps_start=0.995, eps_end=0.005, gamma=0.995, lr=0.001)
```

## Deep Q-Networks (DQN):
```python
from DQN import DQN

dqn = DQN(action_dim=env.action_space.n, observ_dim=env.observation_space.shape[0],
          episodes=EPISODES, lr=0.001, eps_start=0.995, eps_end=0.05, maxlen_of_buffer=50000, batch_size=64, gamma=0.995, TAU=0.0005)
```

## Deep Deterministic Policy Gradient (DDPG):
```python
from DDPG import DDPG

ddpg = DDPG(max_action=0.4, min_action=-0.4,
            action_dim=env.action_space.shape[0], observ_dim=env.observation_space.shape[0],
            Actor_lr=0.0010, Critic_lr=0.0025, batch_size=64, gamma=0.995, TAU=0.0005,
            max_len_of_buffer=100000, mu=0.0, sigma=0.08, theta=0.2)
```

## Proximal Policy Optimization (PPO):
```python
from PPO import PPO

ppo = PPO(
    has_continuous: bool, Action_dim: int, Observ_dim: int,  
    Actor_lr: float = 0.0010, Critic_lr: float = 0.0025, action_scaling: float = None,
    k_epochs: int = 23, policy_clip: float = 0.2, GAE_lambda: float = 0.95,
    gamma: float = 0.995, batch_size: int = 1024, mini_batch_size: int = 512, 
    use_RND: bool = False, beta: int = 0.02)
```

### Small addition about RND and beta parameters for PPO:
`RND` is the *Random-Network-Destilation* approach developed by **OpenAI** to exploring an large-scale environment like **Atari** games, where the states are a pixel images.
More read: [Reinforcement learning with prediction-based rewards](https://openai.com/index/reinforcement-learning-with-prediction-based-rewards/#main)

`Beta` is parameter which we multiplying `RND`'s rewards, because if they'll be too big, our agent will cycle on getting of `intrinsic rewards` of `RND`, instead `extrinsic rewards` of an environment. Beta is needs to don't let `intrinsic rewards` dominate over `extrinsic rewards`.

# An example of exploitation:

## For Q-learning:
```python
def step():
    state = env.reset()[0]

    while True:
        action = Q_learning.get_action(state)

        next_state, reward, done, truncate, _ = env.step(action) 
            
        Q_learning.update_Q_table(state, action, reward, next_state, done)
            
        state = next_state

        if done or truncate:
            break
        
    Q_learning.call_epsilon_decay()

for episode in tqdm(range(EPISODES)):
    step()

env.close()
```

## For SARSA:
```python
def step():
    state = env.reset()[0]

    while True:
        action = SARSA.get_action(state)

        next_state, reward, done, truncate, _ = env.step(action) 
            
        SARSA.update_Q_table(state, action, reward, next_state, done)
            
        state = next_state

        if done or truncate:
            break
        
    SARSA.call_epsilon_decay()

for episode in tqdm(range(EPISODES)):
    step()

env.close()
```

## For DQN:
```python
def step():
    state = env.reset()[0]

    steps = 1

    while True:
        action = DQN.get_action(state)

        next_state, reward, done, truncate, _ = env.step(action) 
            
        DQN.buffer.push([state, action, reward, next_state])

        state = next_state

        steps += 1

        if steps % 5 == 0:
            DQN.education()
            
        elif steps % 50 == 0:
            DQN.soft_update()

        if done or truncate:
            break
        
    DQN.call_epsilon_decay()

for episode in tqdm(range(EPISODES)):
    step()

env.close()
```
## For DDPG:
```python
def step():
    state = env.reset()[0]

    DDPG.reset_OUNoise()

    steps = 1

    while True:
        action = DDPG.get_action(state)

        next_state, reward, done, truncate, _ = env.step(action) 
            
        DDPG.buffer.push([state, action, reward, next_state])

        state = next_state

        steps += 1

        if steps % 5 == 0:
            DDPG.education()
            
        elif steps % 50 == 0:
            DDPG.soft_update()

        if done or truncate:
            break

for episode in tqdm(range(EPISODES)):
    step()

env.close()
```

## For PPO:
```python
def step(episode: int, pbar: object):
    state = env.reset()[0]

    while True:
        action, state_value, log_prob = ppo.get_action(state)

        next_state, reward, done, truncate, _ = env.step(action) 

        ppo.memory.push(state, action, reward, done, state_value, log_prob)
            
        state = next_state

        if done or truncate:
            break
    
    ppo.education()

for episode in tqdm(range(EPISODES)):
    step()

env.close()
```

### ```Graphic``` initialization:
```python
Graphic = Graphic(
    x = 'Episodes',
    y = 'Rewards per episode',
    title = f'Progress of learning in {env.spec.id} by some a RL algorithm',
    hyperparameters={
        ...
        '''
        e. g.
        "lr": ppo.lr,
        "batch_size": ppo.batch_size,
        "mini_batch_size": ppo.mini_batch_size
        '''
    }
)
```
In ```x = ..```, you specify the name of the x-axis; in ```y = ..```, you specify the name of the y-axis; in ```title = ..```, you specify the title of your graph, and the parameter ```hyperparameters = {'key': value}``` is optional. If you do not specify it, it will be ignored by the program. However, if you set ```hyperparameters```, please note that all keys must be strings, and values desirable must be integers of float. You can get yout hyperparameters with ```algorithm_of_RL.hyperparameter_of_RL```, for example: ```PPO.Actor_lr``` or ```PPO.batch_size```.

For updating of grapgic use:
```python
Graphic.update(x, y)
```
Calling every termination state.

For show your final result of learning, use:
```python
Graphic.show()
```

All code is implemented in the PyTorch framework with Python.

References: https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master/