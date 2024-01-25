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


for i in range(250):
    timestep = i // 25
    paddle = (i % 25) // 5
    ball = i % 5
    paddle_y = env.rows

    obs, state = my_env.inverse(i)
    nobs, nstate, reward, done, _ = env.step_env(key, state, 0, params)
    ni = my_env.index(nstate)
    assert ni < 250
    ni_guess = ball + max(paddle - 1, 0) * 5 + min(timestep + 1, env.rows - 1) * 25
    assert ni == ni_guess