from environments.discrete import GridworldEnv, DiscreteStochastic
import numpy as np
from helper import *
from scipy.special import rel_entr

np.set_printoptions(precision=3, suppress=True)

P = random_simplex((10, 10, 4))
R = np.random.random((10, 4))
mu = random_simplex((10,))

env = DiscreteStochastic(P, R, mu)

mu_2 = random_simplex((10,))

print(rel_entr(mu, mu_2).sum())
print(rel_entr(np.array([1, 0, 0]), np.array([0, 1, 0])).sum())