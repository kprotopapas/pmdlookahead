import timeit
import numpy as np
import time
from algorithms.jym import *
from algorithms.policies import *
from jax import numpy as jnp
import jax
import gymnax

from helper import *

from environments.jym import DeepSea, Catch, JDeepsea
from environments.solvable import Deepsea

my_env = Catch(0.99)

env, params = gymnax.make('Catch-bsuite')

key = jax.random.PRNGKey(42)
obs, state = env.reset(key, params)


for s in range(my_env.num_states):
    for a in range(my_env.num_actions):
        obs, state = my_env.inverse(s)
        nobs, nstate, reward, done, _ = env.step_env(key, state, a, params)
        ni = my_env.index(nstate)
        ni_guess = np.argmax(my_env.P[:, s, a])
        r_guess = my_env.R[s, a]

        assert ni_guess == ni
        assert r_guess == reward