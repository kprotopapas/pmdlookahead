import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DiscreteStochastic(gym.Env):
    def __init__(self, P, R, mu):
        self.n_obs = P.shape[0]
        self.n_act = P.shape[2]

        self.P = P
        self.mu = mu
        self.R = R

        self.observation_space = spaces.Discrete(self.n_obs)
        self.action_space = spaces.Discrete(self.n_act)

        self.name = None

        self.reset()
    

    def reset(self):
        self.state = np.random.choice(np.arange(self.n_obs), p=self.mu)


    def step(self, action):
        dist = self.P[:, self.state, action]
        next_state = np.random.choice(np.arange(self.n_obs), p=dist)
        r = self.R[self.state, action]
        old_state = self.state
        self.state = next_state

        return self.state, r, {'r': r, 's': old_state, 'sp': self.state, 'a': action}
    

    def value(self, pi, gamma):
        P_pi = np.sum(pi * self.P, axis=2)
        M = np.identity(self.n_obs) - gamma * P_pi.T
        r_pi = np.sum(pi * self.R, axis=1)

        V = np.linalg.solve(M, r_pi)

        next_V = np.sum(self.P * V.reshape((self.n_obs, 1, 1)), axis=0)
        Q = self.R + gamma * next_V

        return Q
    

    def bellman(self, V, gamma):
        next_V = np.sum(self.P * V.reshape((self.n_obs, 1, 1)), axis=0)
        Q = self.R + gamma * next_V
        TV = np.max(Q, axis=1)
        return TV


class GridworldEnv(DiscreteStochastic):
    def __init__(self, size: int = 4):
        self.size = size

        n_obs = size**2
        n_act = 4
        self.grid = np.arange(n_obs).reshape((size, size))
        self.target = np.random.choice(np.arange(n_obs))

        mu = np.ones((n_obs, )) / (n_obs - 1)
        mu[self.target] = 0

        P = np.zeros((n_obs, n_obs, n_act))
        R = - np.ones((n_obs, n_act))
        for s in range(n_obs):
            for a in range(n_act):
                sp = self.transition(s, a)
                P[sp, s, a] = 1
                R[s, a] *= 1 * (sp != self.target)

        super().__init__(P, R, mu)


    def transition(self, state, action):
        fb, vh = action % 2, action // 2
        amt = 2 * fb - 1
        v = vh * amt
        h = (1 - vh) * amt

        y, x = np.argwhere(self.grid == state)[0]
        new_y, new_x = np.array([y + v, x + h], dtype=np.int8).clip(0, self.size - 1)
        return self.grid[new_y, new_x]
    

    def step(self, action):
        state, action, info = super().step(action)
        info['terminal'] = state == self.target
        return state, action, info
    

    def __str__(self):
        grid_str = self.grid.astype('str')
        y, x = np.argwhere(self.grid == self.target)[0]
        grid_str[y, x] = 'X'

        y, x = np.argwhere(self.grid == self.state)[0]
        grid_str[y, x] = '#'

        return str(grid_str)