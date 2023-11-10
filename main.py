import numpy as np
import argparse
from matplotlib import pyplot as plt
import time

from environments.discrete import *
from baselines.algorithms import *
from helper import random_simplex


if __name__ == '__main__':
    n_agents = 1
    size = 32
    horizon = 25
    gamma = 1. - 1. / horizon
    epochs = 50
    lr = 0.1
    exact = False
    use_wandb = True
    alg = DEQA

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


    P = random_simplex((size, size, n_actions))
    R = random_simplex((size, n_actions))
    mu = random_simplex((size, ))

    P = np.load('runs/latest/P.npy', allow_pickle=True)
    R = np.load('runs/latest/R.npy', allow_pickle=True)
    mu = np.load('runs/latest/mu.npy', allow_pickle=True)

    env = Tabular(P, R, mu, gamma)
    #env = Gridworld(int(np.sqrt(size)), gamma)
    #env = Trap(size, 1, gamma)


    H = 2 * horizon
    M_l = 2 * horizon
    M_t = horizon
    M_f = horizon


    """
    print('Running approx PI')
    pi = PolicyIteration(env, H, n_trajectories=M_l)
    start = time.time()
    info_pi = pi.learn(epochs, exact=exact, use_wandb=use_wandb, verbose=True)
    time_pi = time.time() - start

    print('Running PQA')
    pmd = EQA(env, horizon=H, n_trajectories=M_l)
    start = time.time()
    info_pmd = pmd.learn(epochs, exact=exact, use_wandb=use_wandb, verbose=True, lr=lr)
    time_pmd = time.time() - start

    print('Running 1-PQA')
    pmd = alg(env, horizon=H, M_l=M_l, M_t=M_t, M_f=M_f, d=1)
    start = time.time()
    info_dpqa = pmd.learn(epochs, exact=exact, use_wandb=use_wandb, verbose=True, lr=lr)
    time_dpqa = time.time() - start

    """
    infos = []
    for d in range(5, 15, 5):
        print(f'Running {d}-PQA')
        deep_pmd = alg(env, horizon=H, M_l=M_l, d=d, M_t=M_t, M_f=M_f)
        start = time.time()
        info_dpmd = deep_pmd.learn(epochs, exact=exact, use_wandb=use_wandb, verbose=True, lr=lr)
        infos.append(info_dpmd)
        time_dpmd = time.time() - start
        print(f'{d}-PQA took time: {time_dpmd}')

    

    subs_pi = [i['suboptimality'] for i in info_pi]
    subs_pmd = [i['suboptimality'] for i in info_pmd]
    subs_dpqa = [i['suboptimality'] for i in info_dpqa]

    k = np.arange(epochs)

    plt.yscale('log')
    plt.plot(k, subs_pi, 'r-', label='Approx PI')
    plt.plot(k, subs_pmd, 'b-', label='PQA')
    plt.plot(k, subs_dpqa, 'k-', label='1-PQA')

    colors = ['c', 'y', 'm', 'g']

    for c, info, d in zip(colors, infos, range(5, 25, 5)):
        subs = [i['suboptimality'] for i in info]
        plt.plot(k, subs, f'{c}-', label=f'{d}-PQA')
    
    plt.legend()

    plt.show()
    print()