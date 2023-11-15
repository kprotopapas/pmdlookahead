from algorithms.baselines import DPMD, Algorithm
from algorithms.policies import Policy
from environments.discrete import Tabular
from collections import defaultdict
import numpy as np


class BanditSearch(object):
    def __init__(self, env: Tabular, pi: Policy, depth: int = 1, alpha: float = 1, horizon: int = 10, n_trajectories: int = 10):
        self.debug = True
        self.env = env
        factory = lambda: BanditNode(env, alpha)

        self.nodes = {}
        factory = lambda: Leaf(env, horizon, n_trajectories)
        self.nodes[depth] = defaultdict(factory)

        for d in reversed(range(d - 1)):
            factory = lambda: BanditNode(env, alpha, self.nodes[d + 1])
            self.nodes[d] = defaultdict(factory)

        self.pi = pi


    def sample_trajectory(self, state=None):
        state = state or self.env.state
        node = self.nodes[(state, 0)]
        for d in range(self.depth):
            a = node.explore()



class BanditNode(object):
    def __init__(self, env: Tabular, alpha, children=None):
        self.env = env
        self.values = np.zeros((env.n_act,)) # TODO REPLACE WITH OPTIMISM
        
        self.s = np.zeros_like(self.values)
        self.t = 0
        self.alpha = alpha


    def explore(self):
        self.t += 1
        bonuses = self.alpha * np.sqrt(np.log(self.t) / self.s)
        bonuses = np.where(np.isnan(bonuses), np.inf, bonuses)

        a = np.argmax(self.values) + bonuses
        self.s[a] += 1
        s, r, _ = self.env.step(a)

        return a
    
    def value(self):
        pass



class Leaf(object):
    def __init__(self, env: Tabular, horizon: int, n_trajectories: int):
        self.env = env
        self.horizon = horizon
        self.M_l = n_trajectories


    def explore(self):
        state = self.env.state

        if self.debug:
            V = self.env.V(self.pi)
            return V[state]
        
        V_l = 0

        for _ in range(M_l):
            for t in range(horizon):
                a = self.pi.sample(state)
                r, state, _ = self.env.step(a)
                V_l += r * self.env.gamma**t
        
        V_l /= M_l
        return V_l















class TSPMD(DPMD):
    def __init__(self, env: Tabular, d=2, horizon=10, n_trajectories=10):
        pi = SoftmaxPolicy(env.n_obs, env.n_act)
        
        super().__init__(env, horizon, n_trajectories, pi, f'{d}-TSPMD')


    def sample_trajectory(self, state=None, action=None):
        return super().sample_trajectory(state, action)
    
    def grad(self, V):
        pass


