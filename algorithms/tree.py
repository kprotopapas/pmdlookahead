from algorithms.baselines import DPMD, Algorithm
from algorithms.policies import Policy
from environments.solvable import Tabular
from collections import defaultdict
import numpy as np



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

        self.values[a] = ((self.t - 1) * self.values[a] + rtg) / self.t # MAX OR RTG
        #self.values[a] += 0.01 * (rtg - self.values[a]) EWA OR NORMAL AVG?

        return self.values.max(), rtg



class BanditTS(object):
    def __init__(self, env: Tabular, pi: Policy, max_depth: int = 1,
                 alpha: float = np.sqrt(2), horizon: int = 10, lam: float = 0):
        self.env = env
        self.alpha = alpha
        self.max_depth = max_depth
        self.horizon = horizon
        self.pi = pi
        self.lam = lam

        factory = lambda: BanditNode(env, alpha, max_depth, pi)
        self.nodes = defaultdict(factory)

        factory_leaves = lambda: Leaf(env, pi, max_depth, horizon)
        self.leaves = defaultdict(factory_leaves)


    def sample_trajectory(self, state=None):
        pi = self.pi
        env = self.env
        tree = self
        if state is None:
            state, _ = self.env.reset()
        else:
            self.env.state = state

        states, actions, rewards = [state], [], []
        
        for d in range(self.max_depth - 1):
            node = self.nodes[state]
            node.t += 1
            
            # 1 uniform action then d - 1 optimal actions
            a = np.random.choice(env.n_act) if d == 0 else node.select_action()
            sp, r, _ = self.env.step(a)

            states.append(sp)
            actions.append(a)
            rewards.append(r)

            state = sp
        
        
        rtg = self.env.rtg(rewards)
        leaf = self.leaves[state]
        val, v = leaf.explore()
        
        delta = 0
        tds = []
        for t, (s, sp, a, r) in enumerate(list(zip(states[:-1], states[1:], actions, rewards))[::-1]):
            t = self.max_depth - t - 1
            npp = leaf if t == self.max_depth - 1 else self.nodes[sp]
            n = self.nodes[s]
            
            td = r + self.env.gamma * npp.value - n.values[a] # npp.value or rtg?

            delta = self.env.gamma * self.lam * delta + td

            stepsize = 1 / n.t
            tds.append(td)
            if t == 0:
                pass
            n.values[a] += stepsize * delta

        return delta, rtg[0]



class BanditNode(object):
    def __init__(self, env: Tabular, alpha: float, max_depth: int, pi: Policy):
        self.env = env
        self.values = np.zeros((env.n_act,))
        self.max_depth = max_depth
        self.pi = pi
        
        self.s = np.zeros_like(self.values)
        self.t = 0
        self.alpha = alpha


    def select_action(self):
        biases = self.alpha * np.sqrt(self.t) / self.s # TODO experiment with, sqrt not sqrt, etc...
        biases = np.where(np.isnan(biases), np.inf, biases)

        indices = self.values + biases
        a = np.random.choice(np.flatnonzero(indices == indices.max()))
        self.s[a] += 1

        return a
    
    
    @property
    def value(self):
        return np.max(self.values)



class Leaf(object):
    def __init__(self, env: Tabular, pi: Policy, depth: int, horizon: int):
        self.env = env
        self.horizon = horizon
        self.depth = depth
        self.pi = pi
        self.t = 0
        self.val = 0


    def explore(self):
        state = self.env.state
        begin_state = state
        self.t += 1

        V_l = 0

        for t in range(self.horizon): # TODO maybe reset? 
            a = self.pi.sample(state)
            state, r, _ = self.env.step(a)
            V_l += r * self.env.gamma**t
        
        self.val = ((self.t - 1) * self.val + V_l) / self.t 
        
        return self.val, self.val # RETURN RTG OR V?


    @property
    def value(self):
        return self.val


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