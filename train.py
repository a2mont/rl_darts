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


def loop(alpha, epsilon, gamma, decay, n_experiments, algs, environments, verbose=False, plot=False):
    max_reward = -np.inf
    optimal = None
    rewards = {}
    alg_index = 0
    env = environments[0]
    total_reward = np.zeros([n_algs])
    n_actions = env.action_space.n
    n_states = state_size()
    for Alg in algs:
        alg = Alg(n_actions, n_states,
                  discount=gamma, alpha=alpha, epsilon=epsilon, decay=decay)
        rewards[alg_index] = []
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
                if done:
                    #print("Episode finished after {} timesteps".format(t+1))
                    break
            rewards[alg_index].append(run_reward)
            total_reward[alg_index] += run_reward
            alg.update_epsilon()
            env.close()

        alg_index += 1
        # Plot moving average
        # plt.plot(range(len(rewards)), pd.Series(
        #     rewards).rolling(1000).mean())
        # plt.savefig(f'aaaa_{experiment}')
        # total_reward /= n_experiments
        # reward_t /= n_experiments
        if sum(total_reward) > max_reward:
            max_reward = sum(total_reward)
            optimal = (alpha, epsilon, gamma, decay)
        if verbose:
            print(
                f'alpha: {alpha}, epsilon: {epsilon}, gamma: {gamma}, decay: {decay}, total_reward: {total_reward}')

    if plot:
        for r in rewards:
            plt.plot(range(len(rewards[r])), rewards[r])
            plt.savefig(f'plot_{r}')

    return max_reward, optimal


n_experiments = 100000
T = 100
environments = []
max_reward = [-np.inf, -np.inf, -np.inf, -np.inf]
optimal = [None, None, None, None]

initial_epsilon = 1
min_epsilon = 0.1
opt_decay = (min_epsilon / initial_epsilon)**(1/n_experiments)

environments.append(gym.make('Darts-v0', n_players=1, players_level=[5]))


algs = []
algs.append(sarsa.Sarsa)
algs.append(qlearning.QLearning)
n_algs = len(algs)

alphas = [0.1 * n for n in range(1, 10)]
epsilons = [0.1 * n for n in range(1, 10)]
gammas = [0.1 * n for n in range(1, 10)]
print('Finding alpha...')
for alpha in alphas:
    # finding alpha with default epsilon=0.3
    reward, params = loop(
        alpha, epsilons[2], gammas[8], opt_decay, n_experiments, algs, environments)
    if reward > max_reward[0]:
        max_reward[0] = reward
        optimal[0] = params
    print('.', end='')
print('\nFinding epsilon...')
for epsilon in epsilons:
    # finding alpha with default alpha=0.1
    reward, params = loop(
        alphas[0], epsilon, gammas[8], opt_decay, n_experiments, algs, environments)
    if reward > max_reward[1]:
        max_reward[1] = reward
        optimal[1] = params
    print('.', end='')
print(
    f'\n----------------------------------\n \
    Max reward: {max_reward[0]}, with parameters: {optimal[0]}, \
    while searching alpha\n-------------------------------')
print(
    f'----------------------------------\n \
    Max reward: {max_reward[1]}, with parameters: {optimal[1]}, \
    while searching epsilon\n-------------------------------')

best_alpha = optimal[0][0]
best_epsilon = optimal[1][1]
for gamma in gammas:
    # finding alpha with default alpha=0.1
    reward, params = loop(
        alphas[0], epsilons[2], gamma,  opt_decay, n_experiments, algs, environments)
    if reward > max_reward[2]:
        max_reward[2] = reward
        optimal[2] = params
    print('.', end='')
print(
    f'----------------------------------\n \
    Max reward: {max_reward[2]}, with parameters: {optimal[2]}, \
    while searching gamma with optimal parameters\n-------------------------------')
best_gamma = optimal[2][2]
reward, params = loop(best_alpha, best_epsilon,
                      best_gamma,  opt_decay, n_experiments, algs, environments, plot=True)
if reward > max_reward[3]:
    max_reward[3] = reward
    optimal[3] = params
print('.', end='')
print(
    f'----------------------------------\n \
    Max reward: {max_reward[3]}, with parameters: {optimal[3]}, \
    optimal parameters\n-------------------------------')
