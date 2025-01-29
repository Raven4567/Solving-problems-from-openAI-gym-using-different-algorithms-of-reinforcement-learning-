# Solving-the-problem-from-openAI-gym-using-different-algorithms-of-reinforcement-learning-

This repository is a collection of various reinforcement learning algorithms, ranging from simple Q-learning to PPO (Proximal Policy Optimization), designed for training in OpenAI Gym environments.

For data visualization, the code uses matplotlib charts, with a moving average added for smoother visuals. The original data graph is semi-transparent, while the smoothed graph has a more vivid color.

# Usage:

## Q-learning:
```python
from Q_learning import Q_learning

Q_learning = Q_learning(Action_dim=env.action_space.n, Observ_dim=env.observation_space.n, 
                        episodes=EPISODES, eps=0.995, eps_min=0.005, gamma=0.99, lr=0.001)
```

## SARSA:
```python
from SARSA import SARSA

SARSA = SARSA(Action_dim=env.action_space.n, Observ_dim=env.observation_space.n, episodes=EPISODES, 
              eps=0.995, eps_min=0.005, gamma=0.99, lr=0.001)
```

## Deep Q-Networks (DQN):
```python
from DQN import DQN

DQN = DQN(Action_dim=env.action_space.n, Observ_dim=env.observation_space.shape[0],
          episodes=EPISODES, lr=0.0005, eps=0.95, eps_min=0.05, maxlen_of_buffer=50000, batch_size=64, gamma=0.99, TAU=0.0005)
```

## Deep Deterministic Policy Gradient (DDPG):
```python
from DDPG import DDPG

DDPG = DDPG(max_action=0.4, min_action=-0.4,
            Action_dim=env.action_space.shape[0], Observ_dim=env.observation_space.shape[0],
            Actor_lr=0.0003, Critic_lr=0.0025, batch_size=64, gamma=0.995, TAU=0.0005,
            max_len_of_buffer=100000, mu=0.0, sigma=0.08, theta=0.2)
```

## Proximal Policy Optimization (PPO):
```python
from PPO import PPO

PPO = PPO(
    has_continuous: bool, Action_dim: int, Observ_dim: int,  
    action_scaling: float = None, Actor_lr: float = 0.001, Critic_lr: float = 0.0025, 
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
def step(episode: int, pbar: object):
    state = env.reset()[0]

    reward_per_episode = 0

    while True:
        action = Q_learning.get_action(state)

        next_state, reward, done, truncate, _ = env.step(action) 
            
        pbar.set_description(f"state: {state}, action: {action}, eps: {Q_learning.eps: .3f}, reward: {reward: .2f}, done: {done or truncate}")
            
        Q_learning.update_Q_table(state, action, reward, next_state, done)
            
        state = next_state

        reward_per_episode += reward

        if done or truncate:
            break
        
    Q_learning.call_epsilon_decay()

for episode in (pbar := tqdm(range(EPISODES))):
    step(episode+1, pbar)

env.close()
```

## For SARSA:
```python
def step(episode: int, pbar: object):
    state = env.reset()[0]

    reward_per_episode = 0

    while True:
        action = SARSA.get_action(state)

        next_state, reward, done, truncate, _ = env.step(action) 
            
        pbar.set_description(f"state: {state}, action: {action}, eps: {SARSA.eps: .3f}, reward: {reward: .2f}, done: {done or truncate}")
            
        SARSA.update_Q_table(state, action, reward, next_state, done)
            
        state = next_state

        reward_per_episode += reward

        if done or truncate:
            break
        
    SARSA.call_epsilon_decay()

for episode in (pbar := tqdm(range(EPISODES))):
    step(episode+1, pbar)

env.close()
```

## For DQN:
```python
def step(episode: int, pbar: object):
    state = env.reset()[0]

    steps = 1
    reward_per_episode = 0

    while True:
        action = DQN.get_action(state)

        next_state, reward, done, truncate, _ = env.step(action) 
            
        pbar.set_description(f"state: {state}, action: {action}, eps: {DQN.eps: .3f}, reward: {reward: .2f}, done: {done or truncate}")
            
        DQN.buffer.push([state, action, reward, next_state])

        state = next_state

        reward_per_episode += reward
        steps += 1

        if steps % 5 == 0:
            DQN.education()
            
        if steps % 50 == 0:
            DQN.soft_update()

        if done or truncate:
            break
        
    DQN.call_epsilon_decay()

for episode in (pbar := tqdm(range(EPISODES))):
    step(episode+1, pbar)

env.close()
```
## For DDPG:
```python
def step(episode: int, pbar: object):
    state = env.reset()[0]

    DDPG.reset_OUNoise()

    steps = 1
    reward_per_episode = 0

    while True:
        action = DDPG.get_action(state)

        next_state, reward, done, truncate, _ = env.step(action) 
            
        pbar.set_description(f"state: {state}, action: {action}, reward: {reward: .2f}, done: {done or truncate}")
            
        DDPG.buffer.push([state, action, reward, next_state])

        state = next_state

        reward_per_episode += reward
        steps += 1

        if steps % 5 == 0:
            DDPG.education()
            
        if steps % 50 == 0:
            DDPG.soft_update()

        if done or truncate:
            break

for episode in (pbar := tqdm(range(EPISODES))):
    step(episode+1, pbar)

env.close()
```

## For PPO:
```python
def step(episode: int, pbar: object):
    state = env.reset()[0]

    steps = 1
    reward_per_episode = 0

    while True:
        action, value, log_prob = PPO.get_action(state)

        next_state, reward, done, truncate, _ = env.step(action) 

        state_for_pbar = ','.join(f'{x: .1f}' for x in state)
        action_for_pbar = ','.join(f'{x: .1f}' for x in action)
            
        pbar.set_description(f"state: [{state_for_pbar}], action: [{action_for_pbar}], reward: {reward: .2f}, done: {done or truncate}")

        PPO.memory.push(state, action, reward, done, value, log_prob)
            
        state = next_state

        reward_per_episode += reward
        steps += 1

        if done or truncate:
            break
    
    PPO.education()

for episode in (pbar := tqdm(range(EPISODES))):
    step(episode+1, pbar)

env.close()
```

### ```Graphic``` initialization:
```python
Graphic = Graphic(
    x = 'Episodes',
    y = 'Rewards per episode',
    title = f'Progress of learning in {env.spec.id} by DQN',
    hyperparameters={
        ...
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

All code is implemented in the PyTorch framework.

References: https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master/