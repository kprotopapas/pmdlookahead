import jax
from jax import numpy as jnp
import numpy as np
import timeit
import gymnax

from algorithms.policies import SoftmaxPolicy
from environments.gymnax import JDeepsea
from environments.solvable import Deepsea
from algorithms.gymnax import JEQA
from algorithms.baselines import EQA


size = 4
gamma = 0.99
horizon = size
trajectories = 16
seed = 42

env = JDeepsea(size, gamma)
params = env.default_params

alg = JEQA(env, horizon, trajectories)
key = jax.random.PRNGKey(seed)

observations, actions, rewards = alg.sample_trajectory(key, env, params)
print(len(actions), len(observations))

duration = timeit.timeit(lambda: alg.sample_trajectory(key, env, params), number=1000)

print(f'JEQA trajectory in {duration / 1000:05f} seconds')
print()

Q = alg.estimate_value(key, env, params)

duration = timeit.timeit(lambda: alg.estimate_value(key, env, params), number=100)
print(f'JEQA value in {duration / 100:05f} seconds')
print()


env = Deepsea(size, gamma)
pi = SoftmaxPolicy(env.n_obs, env.n_act)

Q_real = env.Q(pi)

print(f'Error: {np.max(np.abs(Q - Q_real))}')

V = Q.mean(axis=1)
V_real = env.V(pi)


alg = EQA(env, horizon, trajectories)

duration = timeit.timeit(lambda: alg.sample_trajectory(), number=100)
print(f'EQA trajectory in {duration / 100:05f} seconds')
print()

duration = timeit.timeit(lambda: alg.estimate_value(), number=1)
print(f'EQA value in {duration / 1:05f} seconds')
print()