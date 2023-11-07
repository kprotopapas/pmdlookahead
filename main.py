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
    epochs = 500
    lr = 0.1
    exact = False

    n_actions = 8
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


    P = random_simplex((size, size, n_actions))
    R = random_simplex((size, n_actions))
    mu = random_simplex((size,))
    env = DiscreteStochastic(P, R, mu)

    deqa = DEQA(env, gamma=gamma, d=1)
    info = deqa.learn(epochs=500, use_wandb=False, exact=True, lr=lr)
    subs_base = [i['suboptimality'] for i in info]

    all_subs = []
    for d in range(5, 25, 5):
        deqa = DEQA(env, gamma=gamma, d=d)
        info = deqa.learn(epochs=epochs, use_wandb=False, exact=True, lr=lr)
        subs = [i['suboptimality'] for i in info]
        all_subs.append(subs)

    k = np.arange(epochs)

    plt.yscale('log')
    plt.plot(k, subs_base, 'r-')
    
    colors = ['b', 'c', 'y', 'm', 'g']

    for c, s in zip(colors, all_subs):
        plt.plot(k, s, f'{c}-')

    plt.plot(k, gamma**(2 * np.arange(epochs)) * subs_base[0], f'r--')

    for c, l in zip(colors, range(5, 25, 5)):
        plt.plot(k, gamma**(2 * l * np.arange(epochs)) * subs_base[0], f'{c}--')

    plt.ylim(1e-14, 1e2)

    plt.grid(True)
    plt.show()