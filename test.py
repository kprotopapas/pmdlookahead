from environments.discrete import GridworldEnv
import numpy as np
from helper import *

size = 3
gw = GridworldEnv(size=size)

pi = np.random.sample(size=(size**2, 4))
pi = np.tile(np.array([1, 0, 0, 0]), (size**2, 1))
for i in range(size**2):
    pi[i] = simplex_project_sort(pi[i])

Q = gw.value(pi, gamma=0.9)
print(Q)