import gymnasium as gym
from gymnasium import spaces
import numpy as np

from algorithms.policies import Policy


class Tabular(gym.Env):
    def __init__(self, P, R, mu, gamma, name = None):
        self.n_obs = P.shape[0]
        self.n_act = P.shape[2]

        if not np.allclose(P.sum(axis=0), np.ones_like(P.sum(axis=0))):
            raise Exception('Transition probabilities do not sum to 1')
        
        if not np.allclose(mu.sum(), 1):
            raise Exception('Initial state distribution does not sum to 1')

        self.P = P
        self.mu = mu
        self.R = R
        self.gamma = gamma

        self.observation_space = spaces.Discrete(self.n_obs)
        self.action_space = spaces.Discrete(self.n_act)

        self.name = name or f'DiscreteStochastic-{self.n_obs}-{self.n_act}'

        self.samples = 0
        self.reset()
    

    def reset(self, samples=False):
        self.state = np.random.choice(np.arange(self.n_obs), p=self.mu)
        self.samples = 0 if samples else self.samples
        return self.state, {}


    def step(self, action):
        dist = self.P[:, self.state, action]
        next_state = np.random.choice(np.arange(self.n_obs), p=dist)
        r = self.R[self.state, action]
        old_state = self.state
        self.state = next_state
        self.samples += 1

        return self.state, r, {'r': r, 's': old_state, 'sp': self.state, 'a': action}
    

    def V(self, pi: Policy):
        Q = self.value(pi.probas)
        return np.sum(Q * pi.probas, axis=1)

    
    def Q(self, pi: Policy):
        return self.value(pi.probas)


    # Returns the value function of a policy pi (discount factor gamma)
    def value(self, probas):
        gamma = self.gamma
        P_pi = np.sum(probas * self.P, axis=2)
        M = np.identity(self.n_obs) - gamma * P_pi.T
        r_pi = np.sum(probas * self.R, axis=1)

        V = np.linalg.solve(M, r_pi)

        next_V = np.sum(self.P * V[..., np.newaxis, np.newaxis], axis=0)
        Q = self.R + gamma * next_V

        return Q
    

    # Bellman maximisation operator (VI operator)
    def bellman(self, V):
        gamma = self.gamma
        next_V = np.sum(self.P * V[..., np.newaxis, np.newaxis], axis=0)
        Q = self.R + gamma * next_V
        TV = np.max(Q, axis=1)
        return TV
    

    # Finds the optimal value function via policy iteration
    def optimal(self):
        gamma = self.gamma
        Q = np.zeros_like(self.R)

        pi = 1 * (Q == Q.max(axis=1)[..., np.newaxis])
        pi = pi / pi.sum(axis=1)[..., np.newaxis]
        Q_new = self.value(pi)

        while not np.allclose(Q, Q_new):
            Q = Q_new
            pi = 1 * (Q == Q.max(axis=1)[..., np.newaxis])
            pi = pi / pi.sum(axis=1)[..., np.newaxis]
            Q_new = self.value(pi)

        return Q_new
    

    # Finds the optimal value function via value iteration
    def optimal_vi(self):
        gamma = self.gamma
        V = np.ones((self.n_obs, )) / (1 - gamma)
        V_new = self.bellman(V)

        while not np.allclose(V, V_new):
            V = V_new
            V_new = self.bellman(V)
        
        next_V = np.sum(self.P * V_new[..., np.newaxis, np.newaxis], axis=0)
        Q = self.R + gamma * next_V
        return Q


    def copy(self):
        env = Tabular(np.copy(self.P), np.copy(self.R), np.copy(self.mu), self.gamma, self.name)
        env.state = self.state
        env.samples = self.samples
    
        return env

class Trap(Tabular):
    # Defines a Trap environment, where an agent needs to sacrifice immediate reward for large future reward
    # Reward structure looks like: [large, -1, ..., -1, -1, 0, +1, +1, ..., +1, -large]
    def __init__(self, length: int, factor: float, gamma: float):
        size = 2 * length + 1
        n_actions = 3

        P = np.zeros((size, size, n_actions))
        P[:, 1:, 0] = np.identity(size)[:, :-1]
        P[0, 0, 0] = 1

        P[:, :-1, 1] = np.identity(size)[:, 1:]
        P[-1, -1, 1] = 1

        P[:, :, 2] = np.identity(size)

        factor *= 1 / gamma**length

        self.values = np.zeros((size,))
        self.values[0] = factor * length
        self.values[-1] = - factor * length # TODO play with 
        self.values[1:length] = -1
        self.values[length + 1:-1] = 1

        R = np.zeros((size, n_actions))
        R[1:, 0] = self.values[:-1]
        R[0, 0] = self.values[0]
        R[:-1, 1] = self.values[1:]
        R[-1, 1] = self.values[-1]
        R[:, 2] = self.values # NO OP

        R = R / np.max(np.abs(R))

        mu = 1 * (np.arange(size) == length)
        
        name = f'Trap-v2-{length}'
        super().__init__(P, R, mu, gamma, name=name)


    def __str__(self):
        str_values = self.values.astype(str)
        str_values[self.state] = 'X'
        return str(str_values)


class Gridworld(Tabular):
    # Defines a size * size grid with a random target state (absorbing)
    # Reward of -1 for all states except for target
    def __init__(self, size: int, gamma: float):
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
        
        P[:, self.target, :] = 1 * (np.arange(n_obs) == self.target)[..., np.newaxis]
        R[self.target, :] = 0

        name = f'Gridworld-v2-{self.size}'

        super().__init__(P, R, mu, gamma, name=name)


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