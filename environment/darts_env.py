import gym
from gym import spaces
import numpy as np

BOARD_ORDER = [20, 1, 18, 4, 13, 6, 10, 15,
               2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
# 20 rows with multipliers 3x, 1x, 2x, 1x  + bullseyes
DEFAULT_BOARD = [[3, 1, 2, 1]*20, [1, 1, 1, 1], [1, 1, 1, 1]]

N_CELLS = 20*4 + 2


class DartsEnv(gym.Env):

    def __init__(self):

        # Stores thrown darts' positions
        self.darts_positions = [0]*N_CELLS

        # Action: Choose one of the
        self.action_space = spaces.Discrete(N_CELLS)

        # Observation: A board of 20 numbers with 4 multipliers values and 2 bullseyes
        self.observation_space = spaces.Box(
            low=1, high=3, shape=(22, 4), dtype=np.uint8)

    def step(self, action):
        # Execute one time step within the environment
        print('Action :', action)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.darts_positions = [0]*N_CELLS

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print('Render')
