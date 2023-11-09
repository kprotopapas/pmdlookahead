import numpy as np
import argparse
from matplotlib import pyplot as plt

from environments.discrete import *
from baselines.algorithms import *
from helper import random_simplex

if __name__ == '__main__':
    n_agents = 1
    size = 21
    horizon = 10
    gamma = 1. - 1. / horizon
    epochs = 20
    lr = 0.1
    exact = False
    use_wandb = False

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


    env = Trap(size // 2, 1, gamma)

    noise = 0.2

    H = 2 * horizon
    M_l = horizon

    pi = PolicyIteration(env, H, n_trajectories=M_l)
    info = pi.learn(epochs, exact=exact, use_wandb=use_wandb, verbose=True)

    pmd = DPQA(env, horizon=H, M_l=M_l, d=1)
    info_pmd = pmd.learn(epochs, exact=exact, use_wandb=use_wandb, verbose=True, lr=np.inf)

    subs_base = [i['suboptimality'] for i in info]
    subs_pmd = [i['suboptimality'] for i in info_pmd]

    #inaccuracy = [i['normalized_grad_error'] for i in info]
    #print(inaccuracy)

    k = np.arange(epochs)

    plt.yscale('log')
    plt.plot(k, subs_base, 'r-')
    plt.plot(k, subs_pmd, 'b-')
    plt.show()