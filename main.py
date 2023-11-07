import numpy as np
import argparse
from matplotlib import pyplot as plt

from environments.discrete import GridworldEnv, DiscreteStochastic
from baselines.algorithms import REINFORCE, PQA, EQA, DEQA
from helper import random_simplex

if __name__ == '__main__':
    n_agents = 1
    size = 32
    horizon = 100
    env_class = GridworldEnv
    alg = DEQA
    gamma = 1. - 1. / horizon
    epochs = 25
    lr = 1.
    exact = False

    n_actions = 4
    d = 8

    parser = argparse.ArgumentParser(prog='PMD TS runner', description='Runner for PMD TS', epilog='')
    parser.add_argument('--n_agents', '-n', type=int, default=100)
    parser.add_argument('--env_name', '-e', type=str, default='Gridworld')
    parser.add_argument('--horizon', '-t', type=int, default=100)
    parser.add_argument('--alg', '-a', type=str, default='reinforce')
    parser.add_argument('--gamma', '-g', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('-l', type=float, default=0.05)
    parser.add_argument('--exact', action='store_true')


    P = np.load('P.npy', allow_pickle=True)
    R = np.load('R.npy', allow_pickle=True)
    mu = np.load('mu.npy', allow_pickle=True)
    env = DiscreteStochastic(P, R, mu)

    deqa = DEQA(env, gamma=gamma, d=1)
    info = deqa.learn(epochs=500, use_wandb=True, exact=True)
    print()