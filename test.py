from environments.discrete import GridworldEnv, DiscreteStochastic
import numpy as np
from helper import *
from scipy.special import rel_entr

np.set_printoptions(precision=3, suppress=True)

gw = GridworldEnv(4)
print(gw)