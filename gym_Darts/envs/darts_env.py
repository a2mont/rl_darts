import random
import gym
from gym import spaces
import numpy as np

# Aliases to use in code
SEMI_BULLSEYE = 25
BULLSEYE = 50

# Game structure
BOARD_ORDER = [20, 1, 18, 4, 13, 6, 10, 15,
               2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, SEMI_BULLSEYE, BULLSEYE]
# 20 rows with multipliers 3x, 1x, 2x, 1x  + 2 bullseyes *4 for consistency
DEFAULT_BOARD = np.array([[0, 0, 0, 0]]*20 + [[0]*4] + [[0]*4]).tolist()
MULTIPLIERS = [2, 1, 3, 1]
POSSIBLE_SCORES = [x*y for x in BOARD_ORDER[:-2]
                   for y in MULTIPLIERS] + [SEMI_BULLSEYE, BULLSEYE]
POSSIBLE_SCORES = list(dict.fromkeys(POSSIBLE_SCORES))
POSSIBLE_SCORES.sort(reverse=True)
N_CELLS = len(DEFAULT_BOARD) * 4

# Penalties given per:
#   - dart already in the cell
#   - harder shot (3 or 2x and bullseye)
# size of surfaces : https://pages.cs.wisc.edu/~bolo/darts/dartboard.html
DART_PENALTY = 0.01
DOUBLE_PENALTY = 0.05
TRIPLE_PENALTY = 0.075
BULLSEYE_PENALTY = 0.15
SEMI_BULLSEYE_PENALTY = 0.025

# REWARDS
REWARD_OVERFLOW = -10
REWARD_LOSS = -100
REWARD_WIN = 100


# Game rules
INITIAL_SCORE = 501
SHOTS_PER_TURN = 3
MAX_LEVEL = 5  # 5 is the best level of play, 0 the worst
SHOTS_RANGE = 10  # Range of shots selected to calculate the best shot


class DartsEnv(gym.Env):

    def __init__(self, n_players=1, players_level=None, seed=None):

        if seed is not None:
            random.seed(seed)
        # Stores thrown darts' positions
        self.players_score = [INITIAL_SCORE] * n_players
        self.board = create_board()
        self.shot_left = SHOTS_PER_TURN
        if players_level is None:
            self.players_level = [2 for _ in range(n_players)]
        else:
            assert n_players == len(
                players_level), "Size of players and players levels should be the same"
            self.players_level = players_level

        # Action: Choose one of the 22*4 cell to throw at
        self.action_space = spaces.Discrete(N_CELLS)

        # Observation: A board of 20 numbers with 4 multipliers values and 2 bullseyes
        spaces_layers = {
            'board': spaces.Box(low=0, high=3, shape=(22, 4)),
            'scores': spaces.Box(low=0, high=INITIAL_SCORE, shape=(n_players,)),
            'shots': spaces.Discrete(SHOTS_PER_TURN)
        }
        self.observation_space = spaces.Dict(spaces_layers)
        print(
            f'Structure\n'
            f'---------------------------------------------\n'
            f'N cells: {N_CELLS},\n'
            f'Action space: {self.action_space},\n'
            f'Observation space: {self.observation_space}\n'
            f'\nGame rules\n'
            f'---------------------------------------------\n'
            f'N players: {n_players},\n'
            f'Player levels: {self.players_level},\n'
            f'Shots per turn: {self.shot_left},\n'
            f'Initial score: {INITIAL_SCORE},\n')

    def step(self, action, verbose=False):
        # Execute one time step within the environment
        assert action in self.action_space, "Action not in action space"

        if verbose:
            print('Action :', action)

        cell, cell_idx = action_to_cell(action)

        mult_idx = action % 4
        multiplier = MULTIPLIERS[mult_idx] if cell < 25 else 1

        # Register dart position
        self.board[cell_idx][mult_idx] += 1

        score = self.compute_score(cell, multiplier, mult_idx)

        if verbose:
            print(
                f'Mult idx: {mult_idx},Mult: {multiplier}, Cell idx: {cell_idx}, cell: {cell} -> Score: {score}')

        # checks if the player has won (double/BULLSEYE is required to win)
        if self.players_score[0] - score == 0 and (multiplier == 2 or score == BULLSEYE):
            self.players_score[0] = 0

        # Busting ends the turn
        if(self.players_score[0] - score > 1):
            self.players_score[0] -= score
            self.shot_left -= 1
        else:
            self.shot_left = 0

        # Reset the board at the end of the players turn
        if self.shot_left <= 0:
            self.board = create_board()
            self.shot_left = SHOTS_PER_TURN
            # Opponents turn
            self.opponents_turn()

        observation = (self.board, self.players_score, self.shot_left)
        reward = self.compute_reward(score)
        done = any([score == 0 for score in self.players_score])
        info = {}
        return observation, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.board = create_board()
        self.players_score = [INITIAL_SCORE for _ in self.players_score]

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print('Render')

    def compute_score(self, cell, multiplier, mult_idx, verbose=False):
        score = 0
        old_cell = cell
        old_mult = multiplier
        # Each shot has a probability to fail
        miss = random.random()
        direction = random.randint(0, 3)
        # hitting an already thown dart -> the dart does not stick to the board
        if cell == BULLSEYE or cell == SEMI_BULLSEYE:
            dart_nb = sum(self.board[get_cell_index(cell)])
        else:
            dart_nb = self.board[get_cell_index(cell)][mult_idx]
        if miss < dart_nb * DART_PENALTY:
            score = 0
            if verbose:
                print('Missed shot: Dart hit')
            return score

        if cell > SEMI_BULLSEYE:
            if miss < BULLSEYE_PENALTY:
                cell = SEMI_BULLSEYE
                if verbose:
                    print('Missed shot: Bullseye -> Semi-Bullseye')
        elif cell == SEMI_BULLSEYE:
            if miss < SEMI_BULLSEYE_PENALTY:
                while cell == SEMI_BULLSEYE:
                    cell = random.randint(0, len(BOARD_ORDER))
                multiplier = 1
                if verbose:
                    print(f'Missed shot: Semi-Bullseye -> {cell}')
        elif multiplier == 3:
            # direction of the miss 0 -> left, 1 -> right, 2 -> top, 3 -> bottom
            direction = random.randint(0, 3)
            if miss < TRIPLE_PENALTY:
                if direction > 1:
                    multiplier = 1
                    if verbose:
                        print(
                            f'Missed shot: Multpiplier {old_mult} -> {multiplier}')
                else:
                    cell = get_neighbours_cells(
                        cell)[0] if direction == 0 else get_neighbours_cells(cell)[0]
                    if verbose:
                        print(f'Missed shot: Cell {old_cell} -> {cell}')

        elif multiplier == 2:
            if miss < DOUBLE_PENALTY:
                if direction > 1:
                    multiplier = 0 if direction == 2 else 1
                    if verbose:
                        print(
                            f'Missed shot: Multpiplier {old_mult} -> {multiplier}')
                else:
                    cell = get_neighbours_cells(
                        cell)[0] if direction == 0 else get_neighbours_cells(cell)[0]
                    if verbose:
                        print(f'Missed shot: Cell {old_cell} -> {cell}')

        score = multiplier * cell
        return score

    def compute_reward(self, score):
        reward = 0
        player_score = self.players_score[0]
        if player_score == 0:
            reward = REWARD_WIN
            return reward
        if player_score - score < 2:
            reward = REWARD_OVERFLOW
            return reward
        if len(self.players_score) > 1:
            for opp_score in self.players_score[1:]:
                if opp_score == 0:
                    reward = REWARD_LOSS
                    return reward

        return reward

    # Opponenents make the best shot for them, based on their level
    def opponents_turn(self):
        if len(self.players_score) == 0:
            return
        for opp, _ in enumerate(self.players_score[1:]):
            for i in range(3):
                idx = opp + 1
                score = self.find_best_shot(self.players_level[idx], idx, i)

                self.players_score[idx] -= score if self.players_score[idx] - \
                    score > 1 else 0
                # checks if the player has won (double/BULLSEYE is required to win)
                if self.players_score[idx] - score == 0:
                    self.players_score[idx] = 0
                    # End the game
                    return
                # Busting ends the turn
                if(self.players_score[idx] - score > 1):
                    self.players_score[idx] -= score
                else:
                    break

    def find_best_shot(self, level, player_idx, shots):
        # find the best possible shot. If the player is in range of winning then the best shot is the one making them win, otherwise we assume the best shot is
        # the one reducing their score by the largest value
        all_scores = POSSIBLE_SCORES
        if (self.is_winning_range(player_idx)):
            # Winning shot should be a double
            if self.players_score[player_idx] % 2 == 0:
                cell_to_win = self.players_score[player_idx] / 2
                cell_to_win = BULLSEYE if cell_to_win > 20 else cell_to_win
                score = self.compute_score(
                    cell_to_win, 2, MULTIPLIERS.index(2))
                return score
            else:
                # if the player cannot end on a double, it should aim for an even score to end on the next
                all_scores = [x for x in all_scores if x <
                              self.players_score[player_idx] and x % 2 == 1]

        # Opponents also have the possibility to miss
        miss = shots * DART_PENALTY
        if random.random() < miss:
            return 0
        limit = (1 + MAX_LEVEL - level) * SHOTS_RANGE
        shot = random.choice(all_scores[:limit]) if limit < len(
            all_scores) + 1 else random.choice(all_scores + [0])
        return shot

    def is_winning_range(self, player_idx):
        score = self.players_score[player_idx]
        # 50 is the maximal value from one a player can win (requirements: end on a double, 50 = BULLSEYE = 2*SEMI_BULLSEYE)
        return score <= 40 or score == 50


def action_to_cell(action):
    # Gives the correspoding cell (1-20, 25, 50) from the action (0-87)
    index = 1
    cell_idx = 0
    while index <= action:
        if index % 4 == 0:
            cell_idx += 1
        index += 1
    cell = BOARD_ORDER[cell_idx]
    return cell, cell_idx


def get_neighbours_cells(cell):
    # There are no definite neighbours for the center as it is a circle
    if cell == BULLSEYE or cell == SEMI_BULLSEYE:
        return ()
    index = get_cell_index(cell)
    if index >= len(BOARD_ORDER)-2:
        return (BOARD_ORDER[index-1], BOARD_ORDER[0])
    return (BOARD_ORDER[index-1], BOARD_ORDER[index + 1])


def get_cell_index(cell):
    # Find the index of a cell based on the games' ordering
    for i in range(len(BOARD_ORDER)):
        if BOARD_ORDER[i] == cell:
            return i
    return None


def create_board():
    return np.array([[0, 0, 0, 0]]*20 + [[0]*4] + [[0]*4]).tolist()
