import numpy as np
import argparse
from matplotlib import pyplot as plt
import time
import pandas as pd

from environments.discrete import *
from baselines.algorithms import *
from helper import random_simplex


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Euler Batch creator', description='Creates and Runs batch files for euler', epilog='')
    parser.add_argument('--env_name', type=str, default='Gridworld')
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--horizon', '-t', type=int, default=25)
    parser.add_argument('--gamma', '-g', type=float)

    parser.add_argument('--n_agents', '-n', type=int, default=5)
    parser.add_argument('--alg', '-a', type=str, default='DPQA')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', '-l', type=float, default=0.1)
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('-ds', type=[int])

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--group', type=str)

    