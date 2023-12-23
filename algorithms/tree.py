import mctx
import jax
import chex
import numpy as np

from jax import numpy as jnp
from mctx._src.base import PolicyOutput
from algorithms.policies import SoftmaxPolicy

from environments.jym import *
from environments.solvable import *
from planning.jmcts import run_mcts



class PGMCTS(object):
    def __init__(self, policy: JSoftmaxPolicy, depth: int, n_sims: int, M_l: int):
        self.policy = policy
        self.policy_np = policy.numpy()
        self.depth = depth
        self.n_sims = n_sims
        self.M_l = M_l
    

    def learn(self, env: JEnv, lr: float, epochs: int = 1000, callback = None):
        grad = jnp.zeros_like(self.policy.logits)
        
        for k in range(epochs):
            env = env.reset(None)
            key = jax.random.PRNGKey(k)
            action_key, other_key = jax.random.split(key, 2)

            while not env.done:
                output: PolicyOutput = run_mcts(key, env, self.depth, self.policy, self.n_sims, self.M_l)
                grad_k = output.search_tree.children_values[0, 0]
                grad = grad.at[self.policy.embedding(env.state)].add(grad_k)

            
            logits = self.policy.act(env)
            action = jax.random.categorical(key, logits)
            env = env.step(action)

            self.policy = self.policy.update(grad, lr)
            if callback is not None:
                callback(self.policy)
            
            