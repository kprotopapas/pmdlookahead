import mctx
import jax
import chex
import numpy as np
from scipy.stats import entropy
import os


from jax import numpy as jnp
from mctx._src.base import PolicyOutput
from algorithms.baselines import Algorithm
from algorithms.policies import SoftmaxPolicy

from environments.jym import *
from environments.solvable import *
from planning.jmcts import run_mcts


class PGMCTS(Algorithm):
    def __init__(self, env: Tabular, horizon: int = 100, d: int = 1, M_f: int = 1, M_l: int = 128, M_t: int = None):
        
        self.pi = SoftmaxPolicy(env.n_obs, env.n_act)
        self.d = d
        self.n_episodes = M_f
        self.M_l = M_l
        self.simulations = M_t or 2**(d + 1)

        super().__init__(env, horizon, M_f, self.pi, f'{d}-PGMCTS')
        self.config['simulations'] = self.simulations
        self.config['depth'] = self.d
        self.config['episodes'] = self.n_episodes
        self.config['rollouts'] = self.M_l
    

    # Plays a certain number of episodes to estimate the value function
    def estimate_value(self):
        grad = jnp.zeros((self.env.R.shape))
        visits = jnp.zeros((grad.shape[0], ))

        seed = int.from_bytes(os.urandom(8), 'big')

        for k in range(self.n_episodes):
            #key = jax.random.PRNGKey(self.env.samples * (k + 1)) # DETERMINISTIC!!!!!
            key = jax.random.PRNGKey(seed * self.env.samples * (k + 1))

            self.env.reset()
            jenv = self.env.jax()
            jpi = self.pi.jax(jenv.embedding)

            tree_key, action_key = jax.random.split(key)
        
            actions = []
            states = [self.env.state]
            rewards = []

            while not jenv.done:
                # maybe change to fixed horizon rollouts for larger games!!!
                output: PolicyOutput = run_mcts(tree_key, jenv, jpi, self.d, self.simulations, self.M_l)
                tree = output.search_tree
                grad_k = tree.children_rewards[0, 0] + jenv.gamma * tree.children_values[0, 0]

                ind = jnp.dot(jpi.embedding, jenv.state)
                grad = grad.at[ind].add(grad_k)
                visits = visits.at[ind].add(1)

                logits = jpi.act(jenv)
                action_key, _ = jax.random.split(action_key)
                action = jax.random.categorical(action_key, logits)
                jenv, r, _ = jenv.step(action)
                a = int(action)
                s, r, _ = self.env.step(a)

                actions.append(a)
                states.append(s)
                rewards.append(r)


            self.env.samples += self.simulations * self.d

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

        lrk = lr(k) * dk if callable(lr) else lr * dk / self.gamma**(2*(k + 1) * self.d)
        if np.isinf(lrk) or np.isnan(lrk):
            lrk = np.inf
        return lrk, dk
    

    def divergence(self, u, v): # KL-divergence
        return np.max(entropy(u, v, axis=1))