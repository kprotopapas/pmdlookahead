from gymnax import EnvParams
import mctx
import jax
import chex
import numpy as np
from scipy.stats import entropy


from jax import numpy as jnp
from mctx._src.base import PolicyOutput
from algorithms.practical import JAlgorithm
from algorithms.policies import JPolicy

from environments.jym import *
from environments.jym.tabular import JTabular
from environments.solvable import *
from planning.jmcts import run_mcts


class PGMCTS(JAlgorithm):
    def __init__(self, gamma: float, pi: JPolicy, horizon: int = 100, depth: int = 1,
                 M_f: int = 1, M_l: int = 128, M_t: int = None,
                 seed: int = 42):
        
        self.pi = pi
        self.depth = depth
        self.n_episodes = M_f
        self.M_l = M_l
        self.simulations = M_t or 2**(depth + 1)
        self.seed = seed

        super().__init__(gamma, horizon, M_f, self.pi, f'{depth}-PGMCTS')
        self.config['simulations'] = self.simulations
        self.config['depth'] = self.depth
        self.config['episodes'] = self.n_episodes
        self.config['rollouts'] = self.M_l
        self.config['seed'] = self.seed
    

    # Plays a certain number of episodes to estimate the value function
    def estimate_value(self, key: chex.PRNGKey, env: JTabular, params: EnvParams):
        grad = jnp.zeros_like(self.pi.probas)
        visits = jnp.zeros((grad.shape[0], ))

        #seed = int.from_bytes(os.urandom(2), 'big')

        for k in range(self.n_episodes):
            key = jax.random.PRNGKey(self.seed * (k + 1)) # DETERMINISTIC!!!!!
            #key = jax.random.PRNGKey(seed * self.env.samples * (k + 1))

            obs, state = env.reset(params)
            tree_key, action_key = jax.random.split(key)
        
            actions = []
            observations = [obs]
            rewards = []

            done = False
            while not done:
                # maybe change to fixed horizon rollouts for larger games!!!
                output: PolicyOutput = run_mcts(env, params, self.depth, self.simulations, self.M_l, self.gamma,
                                                tree_key, obs, state, self.pi)
                tree = output.search_tree
                grad_k = tree.children_rewards[0, 0] + self.gamma * tree.children_values[0, 0]

                ind = jnp.dot(env.embedding, state)
                grad = grad.at[ind].add(grad_k)
                visits = visits.at[ind].add(1)

                logits = self.pi.act(obs)
                action_key, _ = jax.random.split(action_key)
                action = jax.random.categorical(action_key, logits)
                obs, state, reward, done, _ = env.step(key, state, action, params)

                actions.append(action)
                observations.append(obs)
                rewards.append(reward)


        grad = grad / np.maximum(visits, 1).reshape(-1, 1)
            
        return np.array(grad)


    def grad_exact(self, Q):
        return Q
    

    def lrk(self, lr, k, gradient):
        return lr(k) if callable(lr) else lr, 1
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = self.divergence(greedy, pk)
        if dk == 0: # CONVERGED! V* = TV*
            return 0, 0

        lrk = lr(k) * dk if callable(lr) else lr * dk / self.gamma**(2*(k + 1) * self.depth)
        if np.isinf(lrk) or np.isnan(lrk):
            lrk = np.inf
        return lrk, dk
    

    def divergence(self, u, v): # KL-divergence
        return np.max(entropy(u, v, axis=1))