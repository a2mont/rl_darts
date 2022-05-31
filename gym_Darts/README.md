# Environment

## Build

To build the environment, register the gym environment with

```
pip install -e gym_Darts
```

## Using the environment

To use the environment, import the build from gym

```
import gym
import gym_Darts

env = gym.make('Darts-v0', n_players=#amount_of_players)
```
