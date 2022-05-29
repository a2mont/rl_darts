import random
from time import sleep
import gym
import gym_Darts

env = gym.make('Darts-v0', n_players=5)
env.reset()
end = False
iter = 0
max_iter = 1e6
while not end and iter < max_iter:
    # Random policy
    i = random.randint(0, 87)
    observation, reward, done, info = env.step(i)
    #print(observation, reward, done, info)
    end = done
    iter += 1
    # sleep(1)
print(f'Done in {iter} iterations')
