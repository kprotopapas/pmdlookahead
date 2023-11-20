from algorithms.baselines import DPMD, Algorithm
from algorithms.policies import Policy
from environments.solvable import Tabular
from collections import defaultdict
import numpy as np
from collections import deque



class MCTS(object):
    def __init__(self, env: Tabular, pi: Policy, max_depth: int = 1, alpha: float = np.sqrt(2), horizon: int = 10) -> None:
        self.env = env
        self.alpha = alpha
        self.max_depth = max_depth
        
        factory = lambda: MCTSNode(env, pi, depth=0, alpha=alpha, horizon=horizon, max_depth=max_depth)
        self.roots = defaultdict(factory)


    def sample_trajectory(self, state: int = None):
        if state is None:
            state = self.env.state
        else:
            self.env.state = state

        root = self.roots[state]
        return root.explore()


class MCTSNode(object):
    def __init__(self, env: Tabular, pi: Policy, depth: int = 0, alpha: float = np.sqrt(2), horizon: int = 10,
                 max_depth: int = 1) -> None:
        self.env = env
        self.alpha = alpha
        self.pi = pi
        self.horizon = horizon
        self.depth = depth
        self.max_depth = max_depth

        self.t = 0

        if self.depth + 1 == self.max_depth:
            self.factory = lambda: Leaf(env, pi, depth + 1, horizon)
        else:
            self.factory = lambda: MCTSNode(env, pi, depth + 1, alpha=self.alpha,
                                            horizon=self.horizon, max_depth=self.max_depth)
            
        self.children = defaultdict(self.factory)
        self.s = np.zeros((self.env.n_act, ))
        self.values = np.zeros_like(self.s)


    def explore(self):
        self.t += 1
        biases = self.alpha * np.sqrt(np.sqrt(self.t) / self.s)
        biases = np.where(np.isnan(biases), np.inf, biases)

        indices = self.values + biases
        a = np.random.choice(np.flatnonzero(indices == indices.max()))
        self.s[a] += 1

        state = self.env.state
        sp, r, _ = self.env.step(a)
        V_sp, rtg = self.children[sp].explore()
        rtg = r + self.env.gamma * rtg
        m_rtg = r + self.env.gamma * V_sp
        #self.values[a] += (1 / self.t) * (rtg - self.values[a]) # TODO: EWA or mean? rtg or m_rtg?
        self.values[a] =  ((self.t - 1) * self.values[a] + rtg) / self.t 

        return self.values.max(), rtg




class BanditTS(object):
    def __init__(self, env: Tabular, pi: Policy, max_depth: int = 1, alpha: float = 1, horizon: int = 10):
        self.env = env
        self.alpha = alpha
        
        self.nodes = {}
        factory = lambda: Leaf(env, pi, max_depth, horizon)
        self.nodes[max_depth] = defaultdict(factory)

        self.factories = {d: self.factory(d) for d in range(max_depth)}

        self.max_depth = max_depth
        for d in reversed(range(max_depth)):
            self.nodes[d] = defaultdict(self.factories[d])

    
    def factory(self, d: int):
        return lambda: BanditNode(self.env, self.alpha, d, self.nodes[d + 1])


    def sample_trajectory(self, state=None):
        if state is None:
            state, _ = self.env.reset()
        else:
            self.env.state = state

        node = self.nodes[0][state]
        V_max, rtg = node.explore()
        return V_max, rtg


class BanditNode(object):
    def __init__(self, env: Tabular, alpha: float, depth: int, children: defaultdict, beta=0.1):
        self.env = env
        self.values = np.zeros((env.n_act,)) / (1 - env.gamma)# TODO REPLACE WITH OPTIMISM????
        self.depth = depth
        
        self.s = np.zeros_like(self.values)
        self.t = 0
        self.alpha = alpha
        self.beta = beta
        self.children = children


    def explore(self):
        self.t += 1
        biases = self.alpha * np.sqrt(self.t) / self.s # TODO experiment with, sqrt not sqrt, etc...
        biases = np.where(np.isnan(biases), np.inf, biases)

        indices = self.values + biases
        a = np.random.choice(np.flatnonzero(indices == indices.max()))
        self.s[a] += 1

        state = self.env.state
        sp, r, _ = self.env.step(a)
        V_sp, rtg = self.children[sp].explore()
        rtg = r + self.env.gamma * rtg
        m_rtg = r + self.env.gamma * V_sp
        self.values[a] += (1 / self.t) * (rtg - self.values[a]) # TODO: EWA or mean? rtg or m_rtg?
        #self.values[a] =  ((self.t - 1) * self.values[a] + m_rtg) / self.t 


        return self.values.max(), rtg



class Leaf(object):
    def __init__(self, env: Tabular, pi: Policy, depth: int, horizon: int):
        self.env = env
        self.horizon = horizon
        self.depth = depth
        self.pi = pi
        self.t = 0
        self.value = 0


    def explore(self):
        state = self.env.state
        begin_state = state
        self.t += 1

        V_l = 0

        for t in range(self.horizon): # TODO maybe reset? 
            a = self.pi.sample(state)
            state, r, _ = self.env.step(a)
            V_l += r * self.env.gamma**t
        
        self.value = ((self.t - 1) * self.value + V_l) / self.t 
        
        return self.value, V_l





"""
class TSPMD(DPMD):
    def __init__(self, env: Tabular, d=2, horizon=10, n_trajectories=10):
        pi = SoftmaxPolicy(env.n_obs, env.n_act)
        
        super().__init__(env, horizon, n_trajectories, pi, f'{d}-TSPMD')


    def sample_trajectory(self, state=None, action=None):
        return super().sample_trajectory(state, action)
    
    def grad(self, V):
        pass


"""