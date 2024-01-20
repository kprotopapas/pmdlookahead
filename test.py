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

size = 8
gamma = 0.99

env = DeepSea(size, gamma, deterministic=False)
params = env.params

alg = JEQA(env, 100, 1)
pi = JSoftmaxPolicy.uniform(env.num_states, env.num_actions)

key = jax.random.PRNGKey(42)

obs, state = env.reset(key, params)
priors = jnp.zeros((100, 2)).at[:, 0].set(-jnp.inf)
observations, actions, rewards = alg.sample_trajectory_from_state(key, env, params, obs, state, priors, alg.pi)
for obs in observations:
    print(obs)
    input()