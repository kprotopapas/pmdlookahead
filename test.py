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

size = 4
gamma = 0.99

env = DeepSea(size, gamma, deterministic=False)
opt = env.optimal()

jenv = JDeepsea(size, gamma)
opt_mine = jenv.optimal()


P = env.P
JP = jenv.P


""""
params = env.params

alg = JDEQA(env, size, d=20, M_l=1, M_t=1, M_f=1)
key = jax.random.PRNGKey(42)

r = alg.evaluate(key, env, params, 128, alg.pi)
print(r)


key = jax.random.PRNGKey(43)

duration = timeit.timeit(lambda: alg.evaluate(key, env, params, 128, alg.pi), number=100)

print(f'After JIT compilation, takes {duration / 100} seconds')
"""