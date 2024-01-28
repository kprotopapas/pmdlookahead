import timeit
import numpy as np
import time
from algorithms.jym import *
from algorithms.policies import *
from algorithms.tree import PGMCTS
from jax import numpy as jnp
import jax
import gymnax

from helper import *

from environments.jym import DeepSea, Catch, JDeepsea, Gridworld
from environments.solvable import Deepsea

env = Gridworld(0.99, rows=6, columns=4, deterministic=False, seed=78)
params = env.params
key = jax.random.PRNGKey(42)
obs, state = env.reset(key, params)
print(obs)

while True:
    action = int(input())
    key, subkey = jax.random.split(key)
    obs, state, reward, done, _ = env.step(subkey, state, action, params)
    print(obs)
    print(reward)