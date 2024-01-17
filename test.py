
import jax
import numpy as np
import timeit

from algorithms.policies import SoftmaxPolicy
from environments.jym import JDeepsea
from environments.solvable import Deepsea
from algorithms.practical import JEQA


size = 16
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

print(f'Trajectory in {duration / 1000:05f} seconds')
print()

Q = alg.estimate_value(key, env, params)

duration = timeit.timeit(lambda: alg.estimate_value(key, env, params), number=100)
print(f'Value in {duration / 100:05f} seconds')
print()

env = Deepsea(size, gamma)
pi = SoftmaxPolicy(env.n_obs, env.n_act)

Q_real = env.Q(pi)

print(f'Error: {np.max(np.abs(Q - Q_real))}')