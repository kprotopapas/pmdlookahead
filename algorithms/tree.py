from algorithms.baselines import DPMD
from algorithms.policies import SoftmaxPolicy
from environments.discrete import Tabular
from collections import defaultdict


class Tree(object):
    def __init__(self, depth=1, backup = None, explore = None):
        self.tree = {}


class Node(object):
    def __init__(self, value, n, leaf=False):
        self.children = []

    
    def explore(self):

    


class TSPMD(DPMD):
    def __init__(self, env: Tabular, d=2, horizon=10, n_trajectories=10):
        pi = SoftmaxPolicy(env.n_obs, env.n_act)
        
        super().__init__(env, horizon, n_trajectories, pi, f'{d}-TSPMD')


    def sample_trajectory(self, state=None, action=None):
        return super().sample_trajectory(state, action)
    
    def grad(self, V):
        pass


