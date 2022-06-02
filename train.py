from json import tool
import random
from time import sleep
from turtle import st
from training import sarsa, qlearning
import gym_Darts.envs.darts_env as tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gym
import gym_Darts

LOGS = 'logs/'


def moving_average(x, K):
    T = x.shape[0]
    n = x.shape[1]
    m = int(np.ceil(T / K))
    y = np.zeros([m, n])
    for alg in range(n):
        for t in range(m):
            y[t, alg] = np.mean(x[t*K:(t+1)*K, alg])
    return y


def observation_to_state(observation):
    board, score, throws = observation
    darts = []
    far = 300
    mid = 100
    for idx, cell in enumerate(board):
        if sum(cell) > 0:
            pos = [[tools.BOARD_ORDER[idx], tools.MULTIPLIERS[cell.index(x)]]
                   for x in cell if x > 0]
            pos = [x * y if x < 25 else x for x, y in pos]
            _ = [darts.append(p) for p in pos]
    s = 0
    if score[0] > far:
        s = 102
    elif score[0] > mid:
        s = 101
    else:
        s = score[0]
    state = int(np.prod(darts) * s)
    return state


def state_size():
    # Worst case two darts in 82 and max score
    return (20*4+2)**2 * tools.INITIAL_SCORE


def loop(T, alpha, epsilon, gamma, decay, n_experiments, algs, environments, n_players=1, players_level=None, verbose=False, plot=False):
    max_reward = -np.inf
    optimal = None
    rewards = {}
    differences = {}
    alg_index = 0
    env = environments[0]
    total_reward = np.zeros([len(algs)])
    n_actions = env.action_space.n
    n_states = state_size()
    for Alg in algs:
        alg = Alg(n_actions, n_states,
                  discount=gamma, alpha=alpha, epsilon=epsilon, decay=decay)
        rewards[type(alg).__name__] = []
        differences[type(alg).__name__] = []
        for experiment in range(n_experiments):
            observation = env.reset()
            state = observation_to_state(observation)
            alg.reset(state)
            run_reward = 0
            for _ in range(T):
                # env.render()
                action = alg.act()
                #print(observation, action)
                observation, reward, done, info = env.step(action)
                state = observation_to_state(observation)
                alg.update(action, reward, state)
                run_reward += reward
                differences[type(alg).__name__].append(info['diff_with_best'])
                if done:
                    #print("Episode finished after {} timesteps".format(t+1))
                    break
            rewards[type(alg).__name__].append(run_reward)
            total_reward[alg_index] += run_reward
            alg.update_epsilon()
            env.close()

        alg_index += 1
        if sum(total_reward) > max_reward:
            max_reward = sum(total_reward)
            optimal = (alpha, epsilon, gamma, decay)
        if verbose:
            print(
                f'alpha: {alpha}, epsilon: {epsilon}, gamma: {gamma}, decay: {decay}, total_reward: {total_reward}')

    if plot:
        assert len(
            players_level) == n_players, 'Levels length should be equal to players number'
        fig, axs = plt.subplots(len(rewards.keys()))
        fig.suptitle(
            f'Game with {n_players} players, with levels {players_level}')
        for idx, r in enumerate(rewards):
            axs[idx].plot(range(len(rewards[r])), rewards[r], label=r)
            axs[idx].legend()
        fig.savefig(f'{LOGS}algos_{n_players}players')

    return max_reward, optimal, differences


""" 
Run the experiment procedure, if provided with no parameters field :
    1. find optimal values for alpha, epsilon, gamma
    2. plot the result of the experiment with optimal parameters
If provided with parameters (alpha, epsilon, gamma), run only step 2
"""


def player_experiment(T, decay, n_experiments, algorithms, envs, n_players, players_level, parameters=None, to_file=False):
    f = open(f'{LOGS}logs_{n_players}players.txt', 'w')
    max_reward = [-np.inf, -np.inf, -np.inf, -np.inf]
    optimal = [None, None, None, None]
    if parameters is None:
        print(f'Experiment for ')
        alphas = [0.1 * n for n in range(1, 10)]
        epsilons = [0.1 * n for n in range(1, 10)]
        gammas = [0.1 * n for n in range(1, 10)]
        print('Finding alpha...')
        for alpha in alphas:
            # finding alpha with default epsilon=0.3
            reward, params, _ = loop(
                T, alpha, epsilons[2], gammas[8], decay, n_experiments, algorithms, envs)
            if reward > max_reward[0]:
                max_reward[0] = reward
                optimal[0] = params
            print('.', end='')
        print('')
        if not to_file:
            print(
                f'----------------------------------\n'
                f'Max reward: {max_reward[0]}, with parameters: {optimal[0]}, while searching alpha\n-------------------------------')
        else:
            f.write(f'\n----------------------------------\n'
                    f'Max reward: {max_reward[0]}, with parameters: {optimal[0]}, while searching alpha\n-------------------------------')
        print('Finding epsilon...')
        for epsilon in epsilons:
            # finding alpha with default alpha=0.1
            reward, params, _ = loop(
                T, alphas[0], epsilon, gammas[8], decay, n_experiments, algorithms, envs)
            if reward > max_reward[1]:
                max_reward[1] = reward
                optimal[1] = params
            print('.', end='')
        print('')
        if not to_file:
            print(
                f'----------------------------------\n'
                f'Max reward: {max_reward[1]}, with parameters: {optimal[1]}, while searching epsilon\n-------------------------------')
        else:
            f.write(
                f'\n----------------------------------\n'
                f'Max reward: {max_reward[1]}, with parameters: {optimal[1]}, while searching epsilon\n-------------------------------')

        best_alpha = optimal[0][0]
        best_epsilon = optimal[1][1]

        print('Finding gamma...')
        for gamma in gammas:
            # finding alpha with default alpha=0.1
            reward, params, _ = loop(
                T, alphas[0], epsilons[2], gamma,  decay, n_experiments, algorithms, envs)
            if reward > max_reward[2]:
                max_reward[2] = reward
                optimal[2] = params
            print('.', end='')
        print('')
        best_gamma = optimal[2][2]
        if not to_file:
            print(
                f'----------------------------------\n'
                f'Max reward: {max_reward[2]}, with parameters: {optimal[2]}, while searching gamma with optimal parameters\n-------------------------------')
        else:
            f.write(
                f'\n----------------------------------\n'
                f'Max reward: {max_reward[2]}, with parameters: {optimal[2]}, while searching gamma with optimal parameters\n-------------------------------')
    else:
        best_alpha, best_epsilon, best_gamma = parameters
    reward, params, _ = loop(T, best_alpha, best_epsilon,
                             best_gamma,  decay, n_experiments, algorithms, envs, n_players=n_players, players_level=players_level, plot=True)
    if reward > max_reward[3]:
        max_reward[3] = reward
        optimal[3] = params
    print('.', end='')
    print('')
    if not to_file:
        print(
            f'----------------------------------\n'
            f'Max reward: {max_reward[3]}, with parameters: {optimal[3]}, optimal parameters\n-------------------------------')
    else:
        f.write(
            f'----------------------------------\n'
            f'Max reward: {max_reward[3]}, with parameters: {optimal[3]}, optimal parameters\n-------------------------------')
    f.close()
    return max_reward, optimal


def optimal_score_experiment(T, decay, n_experiments, algorithms, envs, n_players, players_level, parameters=None):
    if parameters is None:
        alpha, epsilon, gamma = (0.1, 0.1, 0.9)
    else:
        alpha, epsilon, gamma, _ = parameters
    _, _, diffs = loop(T, alpha, epsilon,
                       gamma,  decay, n_experiments, algorithms, envs, n_players=n_players, players_level=players_level)
    fig, axs = plt.subplots(len(diffs.keys()))
    for idx, r in enumerate(diffs):
        print(
            f'Average distance to best shot with {r}: {sum(diffs[r])/len(diffs[r])}')
        axs[idx].plot(range(len(diffs[r])), diffs[r], label=r)
        axs[idx].legend()
    fig.savefig(f'{LOGS}algos_diffs')


n_experiments = 400
T = 100
max_players = 6
environments = []

initial_epsilon = 1
min_epsilon = 0.1
opt_decay = (min_epsilon / initial_epsilon)**(1/n_experiments)

environments.append(gym.make('Darts-v0', n_players=1, players_level=[5]))


algs = []
algs.append(sarsa.Sarsa)
algs.append(qlearning.QLearning)

# Experiment on the number of players and their level
for n_players in range(1, max_players):
    players_level = [random.randint(0, 5) for _ in range(n_players)]
    print(n_players, players_level)
    _, params = player_experiment(
        T, opt_decay, n_experiments, algs, environments, n_players, players_level, to_file=True)

optimal_score_experiment(T, opt_decay, n_experiments,
                         algs, environments, 1, [5], parameters=params[-1])
