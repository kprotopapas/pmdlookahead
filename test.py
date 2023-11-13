import numpy as np
from matplotlib import pyplot as plt

from environments.discrete import Tabular, Trap
from algorithms.baselines import DEQA
from helper import random_simplex

if __name__ == '__main__':
    xs = np.arange(500)
    ys = 75 / (1 + np.exp(-0.02 * (xs - 200)))

    plt.plot(xs, ys)
    plt.show()