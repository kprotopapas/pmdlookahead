import jax
from jax import numpy as jnp
import numpy as np
import timeit
import gymnax
import time

from algorithms.policies import SoftmaxPolicy
from environments.gymnax import JDeepsea
from environments.solvable import Deepsea
from algorithms.gymnax import JEQA
from algorithms.baselines import EQA


size = 16
gamma = 0.99
horizon = size
trajectories = 1024
seed = 42

env = JDeepsea(size, gamma)
params = env.default_params

alg = JEQA(env, horizon, trajectories)
key = jax.random.PRNGKey(seed)

Q = alg.estimate_value(key, env, params)

env = Deepsea(size, gamma)
pi = SoftmaxPolicy(env.n_obs, env.n_act)

Q_real = env.Q(pi)

print(f'Q Error: {np.max(np.abs(Q - Q_real[:-1]))}')
print()

V = Q.mean(axis=1)
V_real = env.V(pi)

print(f'V Error: {np.max(np.abs(V - V_real[:-1]))}')
print()