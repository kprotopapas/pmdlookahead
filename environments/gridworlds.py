import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridworldEnv(gym.Env):
    def __init__(self, size: int = 4, horizon: int = 300):
        self.size = size
        self.horizon = horizon
        self.observation_space = spaces.Discrete(size * size)
        self.n_obs = size * size
        self.action_space = spaces.Discrete(4)
        self.target = self.observation_space.sample()
        self.target = 0 # TODO remove
        self.reset()
        self._grid = np.arange(size * size).reshape((size, size))
        self.name = f'Gridworld-{size * size}'

        self.P = np.zeros((size * size, size * size, 4)) # s' | s, a
        self.R = 0 - np.ones((size * size, 4)) # s, a

        for s in range(self.n_obs):
            for a in range(4):
                sp = self.transition(s, a)
                self.P[sp, s, a] = 1
                self.R[s, a] *= (sp != 0)
        
    
        for s in range(self.n_obs):
            for a in range(4):
                sp = self.transition(s, a)


    def transition(self, state, action):
        fb, vh = action % 2, action // 2
        amt = 2 * fb - 1
        v = vh * amt
        h = (1 - vh) * amt

        y, x = self.state_to_pos(state)
        new_x, new_y = np.array([y + v, x + h], dtype=np.int8).clip(0, self.size - 1)
        return self._grid[new_x, new_y]


    def reset(self):
        super().reset()
        self.t = 0
        self.mask = np.ma.masked_where(np.arange(self.n_obs) == self.target, np.ones(self.n_obs)).astype(np.int8)
        self._state = self.observation_space.sample(self.mask)
        return self._state, {'terminal': False, 'target': False}
    

    # actions : 0, 1, 2, 3  <-> left, right, up, down
    def step(self, action):
        self.t += 1
        next_state = self.transition(self._state, action)
        
        self._state = next_state
        target = self._state == self.target
        terminal = target or (self.t >= self.horizon)
        r = 0 if target else -1

        if terminal:
            self.reset()

        return self._state, r, {'terminal': terminal, 'target': target}
    

    def state_to_pos(self, state):
        return np.argwhere(self._grid == state)[0]


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


    def set_state(self, state):
        self.reset()
        self._state = state