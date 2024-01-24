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

size = 2
gamma = 0.99

new_env = DeepSea(size, gamma, deterministic=False)
opt = new_env.optimal()

jenv = JDeepsea(size, gamma)
opt_mine = jenv.optimal()

V_opt = opt_mine.max(axis=1)
reshaped = np.zeros((size, size))
for i in range(size * (size + 1) // 2):
    obs, state = new_env.inverse(i)
    row = state.row
    col = state.column
    reshaped[row, col] = V_opt[i]



indices = np.tril(np.arange(size * size).reshape((size, size))).flatten()
indices = np.hstack((np.array([0]), indices[np.nonzero(indices)], np.array([size**2])))

real_env = Deepsea(size, gamma)
real_opt = real_env.optimal()
real_opt = real_opt[indices]


P_real = real_env.P[indices, :, :][:, indices, :]
P_new = new_env.P
P_old = jenv.P

print(P_new - P_real)
print(P_old - P_real)
pass