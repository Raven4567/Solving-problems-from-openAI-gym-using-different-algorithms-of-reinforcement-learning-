# Solving-the-problem-from-openAI-gym-using-different-algorithms-of-reinforcement-learning-

This repository is a collection of various reinforcement learning algorithms, ranging from simple Q-learning to PPO (Proximal Policy Optimization), designed for training in OpenAI Gym environments.

A unique aspect of my code is that I avoid writing many small methods, which can become confusing. Instead, I often write slightly larger methods. I have also significantly refactored the code and they is standardized now, so that the implementations can now be easily imported as modules and used in other programs. Although this code is primarily created for beginners like myself, I’ve also revised the PPO implementation to combine both continuous and discrete versions.

For data visualization, the code uses matplotlib charts, with a moving average added for smoother visuals. The original data graph is semi-transparent, while the smoothed graph has a more vivid color.

# Usage:

## Q-learning:
```python
from Q_learning import Q_learning

Q_learning = Q_learning(Action_dim=env.action_space.n, 
                        Observ_dim=env.observation_space.n, 
                        episodes=EPISODES, eps=0.995, eps_min=0.005, gamma=0.99, lr=0.001)
```

## SARSA:
```python
from SARSA import SARSA

SARSA = SARSA(Action_dim=env.action_space.n, 
              Observ_dim=env.observation_space.n, episodes=EPISODES, 
              eps=0.995, eps_min=0.005, gamma=0.99, lr=0.001)
```

## Deep Q-Networks (DQN):
```python
from DQN import DQN

DQN = DQN(Action_dim=env.action_space.n, 
          Observ_dim=env.observation_space.shape[0],
          episodes=EPISODES, lr=0.0005, eps=0.95, eps_min=0.05, 
          maxlen_of_buffer=50000, batch_size=64, gamma=0.99, TAU=0.0005)
```

## Deep Deterministic Policy Gradient (DDPG):
```python
from DDPG import DDPG

DDPG = DDPG(max_action=0.4, min_action=-0.4, 

            Action_dim=env.action_space.shape[0], 
            Observ_dim=env.observation_space.shape[0],

            Actor_lr=0.0003, Critic_lr=0.0025, 
            batch_size=64, gamma=0.995, TAU=0.0005,
            max_len_of_buffer=100000, 
            
            mu=0.0, sigma=0.08, theta=0.2)
```

## Proximal Policy Optimization (PPO):
```python
from PPO import PPO

PPO = PPO(
    has_continuous=True, 
    Action_dim=env.action_space.n, 
    Observ_dim=env.observation_space.shape[0],

    action_scaling=2.0, count_of_decay=EPISODES, 
    action_std_init=1.4, action_std_min=0.005,

    Actor_lr=0.0005, Critic_lr=0.0025, batch_size=64,
    gamma=0.99, policy_clip=0.2, k_epochs=7)
```

## A small addition for PPO initialization
It can be either continuous or discrete. If you set ```has_continuous=True```, you must set ```action_scaling```, ```count_of_decay``` (```count_of_decay``` it's count of total calls ```PPO.call_action_std_decay()```, for all training loop, normally he's == count of episodes), ```action_std_init```, and ```action_std_min```. However, if you set ```has_continuous=False```, you don't need to set the aforementioned hyperparameters, you can set the aforementioned parameters to ```None``` or even don't specify, as they will be ignored anyway in ```PPO.__init__()``` 'cause they aren’t used in discrete PPO. You can also specify ```batch_size```, and call the ```PPO.education()``` method every few episodes, and even if you don't specify ```batch_size```, it will default to ```None```, and will using all buffer for update policy.

# An example of exploitation:

## For Q-learning and SARSA:
```python
def step(episode: int, pbar: object):
    state = env.reset()[0]

    reward_per_episode = 0

    while True:
        action = Q_learning.get_action(state)
        #action = SARSA.get_action(state)

        next_state, reward, done, truncate, _ = env.step(action) 
            
        pbar.set_description(f"state: {state}, action: {action}, eps: {Q_learning.eps: .3f}, reward: {reward: .2f}, done: {done or truncate}")
            
        Q_learning.update_Q_table(state, action, reward, next_state, done)
        #SARSA.update_Q_table(state, action, reward, next_state, done)
            
        state = next_state

        reward_per_episode += reward

        if done or truncate:
            break
        
    Q_learning.call_epsilon_decay()
    #SARSA.call_epsilon_decay()

for episode in (pbar := tqdm(range(EPISODES))):
    step(episode+1, pbar)

env.close()
```

## For DQN and DDPG:
```python
def step(episode: int, pbar: object):
    state = env.reset()[0]

    # if using DDPG
    # DDPG.reset_OUNoise()

    steps = 1
    reward_per_episode = 0

    while True:
        action = DQN.get_action(state)
        #action = DDPG.get_action(state)

        next_state, reward, done, truncate, _ = env.step(action) 
            
        pbar.set_description(f"state: {state}, action: {action}, eps: {Q_learning.eps: .3f}, reward: {reward: .2f}, done: {done or truncate}")
            
        DQN.buffer.push([state, action, reward, next_state])
        #DDPG.buffer.push([state, action, reward, next_state])

        state = next_state

        reward_per_episode += reward
        steps += 1

        if steps % 5 == 0:
            DQN.education()
            
        if steps % 50 == 0:
            DQN.soft_update()

        #if steps % 5 == 0:
        #    DDPG.education()
        #    
        #if steps % 50 == 0:
        #    DDPG.soft_update()

        if done or truncate:
            break
        
    DQN.call_epsilon_decay()
    # DDPG not using a epislon-greedy strategy.

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
            
        pbar.set_description(f"state: [{state_for_pbar}], action: [{action_for_pbar}], action_std: {PPO.action_std: .3f}, reward: {reward: .2f}, done: {done or truncate}")

        PPO.Memory.states.append(state)
        PPO.Memory.actions.append(action)
        PPO.Memory.rewards.append(reward)
        PPO.Memory.dones.append(done)
        PPO.Memory.values.append(value)
        PPO.Memory.log_probs.append(log_prob)
            
        state = next_state

        reward_per_episode += reward
        steps += 1

        if done or truncate:
            break
    
    PPO.education()
    PPO.call_action_std_decay() #If you using continuous PPO

for episode in (pbar := tqdm(range(EPISODES))):
    step(episode+1, pbar)

env.close()
```

### A small addition for ```Graphic``` initialization:
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
In ```x = ..```, you specify the name of the x-axis; in ```y = ..```, you specify the name of the y-axis; in ```title = ..```, you specify the title of your graph, and the parameter ```hyperparameters = {'key': value}``` is optional. If you do not specify it, it will be ignored by the program. However, if you set ```hyperparameters```, please note that all keys must be strings, and values can be any type. You can get yout hyperparameters with ```Your_alghoritm.your_hyperparameter```

For updating of grapgic use:
```python
Graphic.update(x, y)
```

For show your final result of learning, use:
```python
Graphic.show()
```

# Results of various algorithms across different environments:
|        Environment        | Q-learning | SARSA  | DQN | DDPG | PPO |
|---------------------------|------------|--------|-----|------|-----|
| Cartpole-v1               |||![DQN - CartPole-v1](Results/DQN_CartPole-v1.png)||![PPO - CartPole-v1](Results/PPO_CartPole-v1.png)|
| Acrobot-v1                |||![DQN - Acrobot-v1](Results/DQN_Acrobot-v1.png)||![PPO - Acrobot-v1](Results/PPO_Acrobot-v1.png)|
| Pendulum-v1               ||||![DDPG - Pendulum-v1](Results/DDPG_Pendulum-v1.png)|![PPO - Pendulum-v1](Results/PPO_Pendulum-v1.png)|
| Blackjack-v1              |![Q-learning - Blackjack-v1](Results/Q_learning-Blackjack-v1.png)||![DQN - Blackjack-v1](Results/DQN_Blackjack-v1.png)||![PPO - Blackjack-v1](Results/PPO_Blackjack-v1.png)|
| Taxi-v3                   |![Q-learning - Taxi-v3](Results/Q-learning_Taxi-v3.png)|![SARSA - Taxi-v3](Results/SARSA_Taxi-v3.png)|![DQN - Taxi-v3](Results/DQN_Taxi-v3.png)||![PPO - Taxi-v3](Results/PPO_Taxi-v3.png)|
| Cliffwalking-v0           |![Q-learning - CliffWalking-v0](Results/Q-learning_Cliffwalking-v0.png)|![SARSA - CliffWalking-v0](Results/SARSA_CliffWalking-v0.png)|![DQN - CliffWalking-v0](Results/DQN_CliffWalking-v0.png)||![PPO - CliffWalking-v0](Results/PPO_CliffWalking-v0.png)|
| FrozenLake-v1             |![Q-learning - FrozenLake-v1](Results/Q-learning_FrozenLake-v1.png)|![SARSA - FrozenLake-v1](Results/SARSA_FrozenLake-v1.png)|![DQN - FrozenLake-v1](Results/DQN_FrozenLake-v1.png)||![PPO - FrozenLake-v1](Results/PPO_FrozenLake-v1.png)|
| InvertedDoublePendulum-v5 ||||![DDPG - InvertedDoublePendulum-v5](Results/DDPG_InvertedDoublePendulum-v5.png)|![PPO - InvertedDoublePendulum-v5](Results/PPO_InvertedDoublePendulum-v5.png)|
| InvertedPendulum-v5       ||||![DDPG - InvertedPendulum-v5](Results/DDPG_InvertedPendulum-v5.png)|![PPO - InvertedPendulum-v5](Results/PPO_InvertedPendulum-v5.png)|
| Ant-v5                    ||||![DDPG - Ant-v5](Results/DDPG_Ant-v5.png)|![PPO - Ant-v5](Results/PPO_Ant-v5.png)|
| Hopper-v5                 ||||![DDPG - Hopper-v5](Results/DDPG_Hopper-v5.png)|![PPO - Hopper-v5](Results/PPO_Hopper-v5.png)|
| Humanoid-v5               ||||![DDPG - Humanoid-v5](Results/DDPG_Humanoid-v5.png)|![PPO - Humanoid-v5](Results/PPO_Humanoid-v5.png)|
| HumanoidStandup-v5        ||||![DDPG - HumanoidStandup-v5](Results/DDPG_HumanoidStandup-v5.png)|![PPO - HumanoidStandup-v5](Results/PPO_HumanoidStandup-v5.png)|
| Pusher-v5                 ||||![DDPG - Pusher-v5](Results/DDPG_Pusher-v5.png)|![PPO - Pusher-v5](Results/PPO_Pusher-v5.png)|
| Reacher-v5                ||||![DDPG - Reacher-v5](Results/DDPG_Reacher-v5.png)|![PPO - Reacher-v5](Results/PPO_Reacher-v5.png)|
| Swimmer-v5                ||||![DDPG - Swimmer-v5](Results/DDPG_Swimmer-v5.png)|![PPO - Swimmer-v5](Results/PPO_Swimmer-v5.png)|
| Walker2d-v5               ||||![DDPG - Walker2d-v5](Results/DDPG_Walker2d-v5.png)|![PPO - Walker2d-v5](Results/PPO_Walker2d-v5.png)|


All code is implemented in the PyTorch framework.

References: https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master, GPT-4o and GPT-4o mini (just a little).
