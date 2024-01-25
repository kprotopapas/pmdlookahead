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

from environments.jym import DeepSea, Catch, JDeepsea
from environments.solvable import Deepsea

env = Catch(0.99, rows=32, columns=8)

alg = PGMCTS(env, 32, 5, M_f=1, M_l=128, M_t=50)
key = jax.random.PRNGKey(42)
duration = timeit.timeit(lambda: alg.sample_trajectory(key, env, env.params, alg.pi), number=10)
print(f'Time: {duration / 10}')

alg = JDEQA(env, 32, d=5, M_l=4, M_t=4, M_f=4)
duration = timeit.timeit(lambda: alg.sample_trajectory(key, env, env.params, alg.pi), number=10)

print(f'Time: {duration / 10}')