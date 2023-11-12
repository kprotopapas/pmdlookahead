import numpy as np
import argparse
from matplotlib import pyplot as plt
import time
from threading import Thread
import pandas as pd

from environments.discrete import *
from baselines.algorithms import *
from helper import random_simplex

def run(d=1):
    print(f'Running {alg.__name__} with depth {d}')
    deep_pmd = alg(env, horizon=H, M_l=M_l, d=d, M_t=M_t, M_f=M_f)

    start = time.time()
    info_dpmd = deep_pmd.learn(epochs, exact=exact, use_wandb=use_wandb, verbose=True, lr=lr, group=group)
    time_dpmd = time.time() - start
    print(f'{alg.__name__} with depth {d} took time {time_dpmd}')

    info_df = pd.DataFrame.from_records(info_dpmd)
    info_df.set_index('epoch', drop=True)
    return info_df


def run_thread(d=1):
    daemon = Thread(target=run, kwargs={'d': d})
    daemon.daemon = False
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
    parser.add_argument('--env_name', type=str, default='Gridworld')
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--horizon', '-t', type=int, default=25)
    parser.add_argument('--gamma', '-g', type=float)

    parser.add_argument('--n_agents', '-n', type=int, default=5)
    parser.add_argument('--alg', '-a', type=str, default='DPQA')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', '-l', type=float, default=0.1)
    parser.add_argument('--exact', action='store_true')

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--thread', action='store_true')
    parser.add_argument('--group', type=str)


    args = parser.parse_args()

    n_actions = 4
    size = args.size
    horizon = args.horizon
    gamma = args.gamma or 1 - 1 / horizon
    assert gamma >= 0 and gamma < 1

    use_wandb = args.wandb
    thread = args.thread
    group = args.group

    alg = parse_alg(args.alg)
    n_agents = args.n_agents
    exact = args.exact
    lr = args.lr
    epochs = args.epochs

    env = Trap(size, 1, gamma)

    H = 2 * horizon
    M_l = 2 * horizon
    M_t = horizon
    M_f = horizon

    infos = []
    ds = [1]
    ds.extend(range(5, 5 * n_agents, 5))
    
    # TRY WITH LOGISTIC LEARNING RATE

    a = 75
    lr = lambda k: a / (1 + np.exp(-10 * (k - epochs / 3) / epochs))
    for d in ds:
        if thread:
            run_thread(d=d)
        else:
            infos.append(run(d=d))

    if not thread:
        for df in infos:
            df.suboptimality.plot()

        plt.yscale('log')
        plt.show()
    