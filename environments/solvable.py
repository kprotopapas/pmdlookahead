import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnax.environments.environment import Environment

from algorithms.policies import Policy
from environments.gymnax import JDeepsea, JTrap


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


    def sample_trajectory(self, pi: Policy, horizon: int):
        states, actions, rewards = [self.state], [], []
        for t in range(horizon):
            a = pi.sample(self.state)
            sp, r, _ = self.step(a)
            states.append(sp)
            actions.append(a)
            rewards.append(r)
        

        return states, actions, rewards


    def rtg(self, rewards):
        discounts = self.gamma**np.arange(len(rewards))
        return np.cumsum((discounts * rewards)[::-1])[::-1].copy()
    

    def jax(self) -> Environment:
        pass


class Trap(Tabular):
    # Defines a Trap environment, where an agent needs to sacrifice immediate reward for large future reward
    # Reward structure looks like: [large, -1, ..., -1, -1, 0, +1, +1, ..., +1, -large] (then normalized to 1)
    def __init__(self, size: int, gamma: float, noop: bool = False):
        self.size = size
        length = 2 * size + 1
        n_actions = 3 if noop else 2

        P = np.zeros((length, length, n_actions))
        P[:, 1:, 0] = np.identity(length)[:, :-1]
        P[0, 0, 0] = 1

        P[:, :-1, -1] = np.identity(length)[:, 1:]
        P[-1, -1, -1] = 1

        if noop:
            P[:, :, 1] = np.identity(length)

        factor = 0.5 * gamma**size * (1 - gamma)
        self.factor = factor

        self.values = np.zeros((length,))
        self.values[0] = 1
        self.values[1:size] = -factor
        self.values[size + 1:] = factor

        R = np.zeros((length, n_actions))
        R[1:, 0] = self.values[:-1]
        R[0, 0] = self.values[0]
        R[:-1, -1] = self.values[1:]
        R[-1, -1] = self.values[-1]

        if noop:
            R[:, 1] = self.values # NO OP

        R = R / np.max(np.abs(R))

        mu = 1 * (np.arange(length) == size)
        
        name = f'Trap-v2-{size}'
        super().__init__(P, R, mu, gamma, name=name)


    def jax(self) -> Environment:
        return JTrap(self.size, self.gamma)


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


class Deepsea(Tabular):
    # Deepsea environment from bsuite
    def __init__(self, size: int, gamma: float):
        self.size = size

        n_obs = size**2 + 1
        n_act = 2 # go left or right
        self.grid = np.arange(size**2).reshape((size, size))

        mu = np.zeros((n_obs, ))
        mu[0] = 1

        P = np.zeros((n_obs, n_obs, n_act))
        R = np.zeros((n_obs, n_act))
        R[:, 1] = -0.1 * gamma**size / size # TODO penalty different to bsuite...
        R[-2, 1] = 1
        R[-1, :] = 0

        for s in range(n_obs):
            for a in range(n_act):
                sp = self.transition(s, a)
                P[sp, s, a] = 1

        name = f'Deepsea-v2-{self.size}'

        super().__init__(P, R, mu, gamma, name=name)


    def transition(self, state, action):
        if state == self.size**2:
            return state
        
        pos = np.argwhere(self.grid == state)[0]
        dx = 2 * action - 1
        delta = np.where(pos[0] == self.size - 1, np.zeros((2,), dtype=np.int32), np.array([1, dx], dtype=np.int32))

        new_y, new_x = np.clip(pos + delta, 0, self.size - 1)

        if pos[0] == self.size - 1:
            return self.size**2
        
        return self.grid[new_y, new_x]
    

    def step(self, action):
        state, action, info = super().step(action)
        info["terminal"] = state == self.n_obs - 1
        return state, action, info
    

    def __str__(self):
        grid_str = self.grid.astype('str')
        grid_str[-1, -1] = 'X'

        if self.state != self.size**2:
            y, x = np.argwhere(self.grid == self.state)[0]
            grid_str[y, x] = '#'

        return str(grid_str)


    def jax(self) -> Environment:
        return JDeepsea(self.size, self.gamma)
    

class Cliffwalk(Tabular):
    # Cliffwalk environment from Sutton & Barto: Reinforcement Learning, an Introduction
    def __init__(self, width: int, gamma: float, height: int = 4):
        self.width = width
        self.height = height

        n_obs = width * height
        n_act = 4
        self.grid = np.arange(n_obs).reshape((height, width))
        self.start = self.grid[0, 0]
        self.target = self.grid[0, -1]

        mu = 1 * (np.arange(n_obs) == self.start)

        P = np.zeros((n_obs, n_obs, n_act))
        R = - np.ones((n_obs, n_act))
        for s in range(n_obs):
            for a in range(n_act):
                sp, fall = self.transition(s, a)
                P[sp, s, a] = 1
                y, x = np.argwhere(self.grid == sp)[0]
                if fall:
                    R[s, a] = -100
        
        
        P[:, self.target, :] = 1 * (np.arange(n_obs) == self.target)[..., np.newaxis] # target is absorbing
        R[self.target, :] = 0
        
        R = R / np.max(np.abs(R))

        name = f'Cliffwalk-{self.width}x{self.height}'

        super().__init__(P, R, mu, gamma, name=name)


    def transition(self, state, action):
        fb, vh = action % 2, action // 2
        amt = 2 * fb - 1
        v = vh * amt
        h = (1 - vh) * amt

        y, x = np.argwhere(self.grid == state)[0]
        new_y = min(self.height - 1, max(y + v, 0))
        new_x = min(self.width - 1, max(x + h, 0))

        if new_y == 0 and new_x > 0 and new_x < self.width - 1:
            return self.start, True
        
        return self.grid[new_y, new_x], False
    

    def step(self, action):
        state, action, info = super().step(action)
        info['terminal'] = state == self.target
        return state, action, info
    

    def __str__(self):
        grid_str = self.grid.astype('str')
        y, x = np.argwhere(self.grid == self.target)[0]
        grid_str[y, x] = 'X'

        grid_str[0, 1:-1] = '^'

        y, x = np.argwhere(self.grid == self.state)[0]
        grid_str[y, x] = '#'


        return str(grid_str)
