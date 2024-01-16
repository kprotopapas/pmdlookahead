import time
import chex
import jax
from jax import numpy as jnp
from environments.jym import JDeepsea
from algorithms.practical import JEQA

import timeit

env = JDeepsea(16, 0.99)
params = env.default_params

alg = JEQA(env, 16, 2)

key = jax.random.PRNGKey(42)

def step(key: chex.PRNGKey, state: EnvState):