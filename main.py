import numpy as np
import argparse
from matplotlib import pyplot as plt
import time
from threading import Thread

from environments.discrete import *
from baselines.algorithms import *
from helper import random_simplex

def run(d=1):
    print(f'Running {alg.__name__} with depth {d}')
    deep_pmd = alg(env, horizon=H, M_l=M_l, d=d, M_t=M_t, M_f=M_f)
    start = time.time()
    info_dpmd = deep_pmd.learn(epochs, exact=exact, use_wandb=use_wandb, verbose=True, lr=lr)
    time_dpmd = time.time() - start
    print(f'{alg.__name__} with depth {d} took time {time_dpmd}')
    return info_dpmd


def run_thread(d=1):
    daemon = Thread(target=run, kwargs={'d': d})
    daemon.daemon = True
    daemon.start()


def parse_alg(alg: str):
    match alg.upper():
        case 'REINFORCE':
            return REINFORCE
        case 'DPQA':
            return DPQA
        case 'DEQA':
            return DEQA
        case 'PQA':
            return PQA
        case 'EQA':
            return EQA


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PMD TS runner', description='Runner for PMD TS', epilog='')
    parser.add_argument('--n_agents', '-n', type=int, default=5)
    parser.add_argument('--env_name', '-e', type=str, default='Gridworld')
    parser.add_argument('--horizon', '-t', type=int, default=25)
    parser.add_argument('--alg', '-a', type=str, default='DPQA')
    parser.add_argument('--gamma', '-g', type=float)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-l', type=float, default=0.1)
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--size', '-s', type=int, default=32)

    args = parser.parse_args()

    n_agents = args.n_agents
    size = 32
    n_actions = 4


    horizon = args.horizon
    gamma = args.gamma if 'gamma' in args else 1 - 1 / horizon
    epochs = args.epochs
    lr = args.lr

    exact = args.exact
    use_wandb = args.wandb

    alg = parse_alg(args.alg)


    P = random_simplex((size, size, n_actions))
    R = random_simplex((size, n_actions))
    mu = random_simplex((size, ))

    #env = Tabular(P, R, mu, gamma)
    #env = Gridworld(int(np.sqrt(size)), gamma)
    env = Trap(size, 1, gamma)


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
    """
    
    lr = lambda k: 0.1 / gamma**(2 * (k + 1) * 1)
    run_thread(d=1)

    infos = []
    for d in range(5, 5 * n_agents, 5):
        run_thread(d=d)