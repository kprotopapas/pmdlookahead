import timeit
import numpy as np
import time
from algorithms.gymnax import *
from algorithms.policies import *
from jax import numpy as jnp
import jax

from helper import *

from environments.gymnax import JDeepsea
from environments.solvable import Deepsea

size = 16
gamma = 0.99

env = JDeepsea(size, gamma)
params = env.default_params
alg = JDPQA(env, size, 16, 1, 1, 1)

grad = jnp.zeros_like(alg.pi.probas)
grad = grad.at[-2].set(jnp.array([0, 1]))
alg.pi = alg.pi.update(grad, 1)
print(alg.pi.probas[-2])


key = jax.random.PRNGKey(42)
value_key, sample_key = jax.random.split(key)

V_est = alg.estimate_value(value_key, env, params)
V = env.V(alg.pi)

print(V_est[-2])
print(V[-2])