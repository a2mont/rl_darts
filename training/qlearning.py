import numpy as np


class QLearning:
    def __init__(self, n_actions, n_states, discount=0.9, alpha=0.01, epsilon=0.1, decay=0, min_epsilon=0.1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.Q = np.zeros([n_states, n_actions])
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay = decay
        self.state = 0
        self.min_epsilon = min_epsilon

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay

    def act(self):
        # by default, act greedily
        if (np.random.uniform() < self.epsilon):
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[self.state, :])

    def update(self, action, reward, state):
        # fill in
        Q_max = np.max(self.Q[state, :])
        self.Q[self.state, action] = self.alpha * \
            (reward + self.discount * Q_max) + \
            (1.0 - self.alpha) * self.Q[self.state, action]
        self.state = state

    def reset(self, state):
        self.state = state
