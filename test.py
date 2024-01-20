import timeit
import numpy as np
import time
from algorithms.jym import *
from algorithms.policies import *
from jax import numpy as jnp
import jax
import gymnax

from helper import *

from environments.jym import JDeepsea
from environments.jym import DeepSea
from environments.solvable import Deepsea

size = 16
gamma = 0.99

env, params = gymnax.make('DeepSea-bsuite', size=8)
print(params)
params = params.replace(deterministic=False)

env = DeepSea(size, gamma, deterministic=False)
params = env.params

alg = JEQA(env, 100, 1)
embedding = jnp.ravel
pi = JSoftmaxPolicy.uniform(env.num_states, env.num_actions, embedding=embedding)

key = jax.random.PRNGKey(42)
observations, actions, rewards = alg.sample_trajectory(key, env, params, alg.pi)

pass