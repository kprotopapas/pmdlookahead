import timeit
import numpy as np
import time
from algorithms.gymnax import JEQA
from algorithms.policies import JSoftmaxPolicy
from jax import numpy as jnp
import jax

from environments.gymnax import JDeepsea
from environments.solvable import Deepsea

size = 16
gamma = 0.99
start = time.time()
env = JDeepsea(size, gamma)
end = time.time()

key = jax.random.PRNGKey(42)
params = env.default_params

alg = JEQA(env, size, 1)

t = alg.sample_trajectory(key, env, params)

duration = timeit.timeit(lambda: alg.sample_trajectory(key, env, params), number=1000)

print(f'Sample trajectory now takes {duration / 1000} seconds')

Q = alg.estimate_value(key, env, params)
duration = timeit.timeit(lambda: alg.estimate_value(key, env, params), number=100)
print(f'Estimate value takes {duration / 100} seconds')