from environments.discrete import GridworldEnv, DiscreteStochastic
import numpy as np
from helper import *

np.set_printoptions(precision=3, suppress=True)

P = random_simplex((10, 10, 4))
R = np.random.random((10, 4))
mu = random_simplex((10,))

env = DiscreteStochastic(P, R, mu)
print(env.R)
print(env.step(1))
print(env.optimal(0.9))