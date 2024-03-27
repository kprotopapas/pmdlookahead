from gymnax import EnvParams, EnvState
import mctx
import jax
import chex
import numpy as np
from scipy.stats import entropy
from functools import partial


from jax import numpy as jnp
from mctx._src.base import PolicyOutput
from algorithms.jym import JAlgorithm
from algorithms.policies import JPolicy, JSoftmaxPolicy

from environments.jym import *
from environments.jym.tabular import JTabular
from environments.solvable import *
from planning.jmcts import run_mcts


class PGMCTS(JAlgorithm):
    def __init__(self, env: JTabular, horizon: int = 100, depth: int = 1,
                 M_f: int = 1, M_l: int = 128, M_t: int = None):
        
        self.pi = JSoftmaxPolicy.uniform(env.num_states, env.num_actions)
        self.depth = depth
        self.n_episodes = M_f
        self.M_l = M_l
        self.simulations = M_t or 2**(depth + 1)

        super().__init__(env, horizon, M_f, self.pi, f'{depth}-PGMCTS')
        self.config['simulations'] = self.simulations
        self.config['depth'] = self.depth
        self.config['episodes'] = self.n_episodes
        self.config['rollouts'] = self.M_l


    @partial(jax.jit, static_argnums=(0, 2))
    def sample_trajectory_from_state(self, key: chex.PRNGKey, env: JTabular, params: EnvParams,
                          obs: chex.Array, state: EnvState, priors: chex.Array, pi: JPolicy):
        
        def step(a: tuple[chex.PRNGKey, chex.Array, EnvState, jnp.bool_], prior: chex.Array):
            key, obs, state, old_done = a
            new_key, env_key, action_key = jax.random.split(key, 3)
            ind = env.index(state)
            logits = pi.act(ind)
            action = jax.random.categorical(action_key, logits + prior)
            nobs, nstate, reward, done, _ = env.step(env_key, state, action, params)
            reward = jnp.where(old_done, 0, reward)
            return (new_key, nobs, nstate, done | old_done), (ind, action, reward)


        carry, ys = jax.lax.scan(step, (key, obs, state, jnp.bool_(False)), priors, length=self.horizon)

        nkey, nobs, nstate, done = carry
        indices, actions, rewards = ys
        index = env.index(nstate)
        indices = jnp.append(indices, index)
        return indices, actions, rewards
    
    

    # Plays a certain number of episodes to estimate the value function
    def estimate_value(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, pi: JPolicy):
        traj_key, tree_key = jax.random.split(key)
        indices, actions, rewards = self.sample_trajectory(traj_key, env, params, pi)
        
        keys = jax.random.split(tree_key, len(indices))
        gradients = []
        for (i, sub_key) in zip(indices, keys):
            obs, state = env.inverse(i)
            output: PolicyOutput = run_mcts(env, params, self.depth, self.simulations, self.M_l,
                                            env.gamma, sub_key, obs, state, pi)
            tree = output.search_tree
            grad_k = tree.children_rewards[0, 0] + env.gamma * tree.children_values[0, 0]
            gradients.append(grad_k)

        gradients = jnp.vstack(gradients)
        
        
        Q = jnp.zeros_like(self.pi.probas).at[indices].set(gradients)
        return Q, self.horizon * self.simulations * self.depth * self.M_l


    def grad_exact(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, Q: chex.Array, pi: JPolicy):
        return Q
    

    def lrk(self, lr, k, gradient):
        lrk = lr(k) if callable(lr) else lr
        return lrk, 0
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