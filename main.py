import numpy as np
import argparse

from environments.discrete import GridworldEnv, DiscreteStochastic
from baselines.algorithms import REINFORCE, PQA, EQA, DEQA
from helper import random_simplex

if __name__ == '__main__':
    n_agents = 1
    size = 8
    horizon = 100
    env_class = GridworldEnv
    alg = DEQA
    gamma = 0.99
    epochs = 500
    lr = lambda k: 1. / gamma**(2 * (k + 1) * d)
    exact = True

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

    
    best = np.inf

    for agent in range(n_agents):
        P = random_simplex((size, size, n_actions))
        R = np.random.random((size, n_actions))
        mu = random_simplex((size, ))
        env = DiscreteStochastic(P, R, mu)

        # algorithm = DEQA(env, gamma=gamma, horizon=horizon, d=d)
        # perf = algorithm.learn(epochs=epochs, lr=lr, exact=True)

        algorithm = EQA(env, gamma=gamma, horizon=10, n_trajectories=100)
        perf = algorithm.learn(epochs=epochs, lr=lr, exact=False)

        if sum(perf[-10:]) < best:
            best = sum(perf[-10:])
            algorithm.pi.save(f'weights/Gridworld-{size}-{alg.__name__}.npy')