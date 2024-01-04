import numpy as np
import argparse
from matplotlib import pyplot as plt
import time
import pandas as pd
from algorithms.tree import PGMCTS

from environments.jym import JDeepsea

from environments.solvable import *
from algorithms.baselines import *
from helper import random_simplex, parse_args

def run(d=1):
    if alg != EQA:
        agent = alg(env, horizon=H, d=d, M_t=M_t) # time per iteration 0.3 s
    else:
        agent = alg(env, horizon=H, n_trajectories=1) # time per iteration 0.008 s
        
    start = time.time()
    info_dpmd = agent.learn(epochs, exact=exact, use_wandb=use_wandb, verbose=True, lr=lr, group=group)
    time_dpmd = time.time() - start
    print(f'{alg.__name__} with depth {d} took time {time_dpmd}')

    info_df = pd.DataFrame.from_records(info_dpmd)
    info_df.set_index('epoch', drop=True)
    return info_df


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
        case 'PGMCTS':
            return PGMCTS


if __name__ == '__main__':
    args = parse_args()

    size = args.size
    horizon = args.horizon
    gamma = args.gamma or 1 - 1 / horizon
    assert gamma >= 0 and gamma < 1

    use_wandb = args.wandb
    group = args.group

    alg = parse_alg(args.alg)
    n_agents = args.n_agents
    exact = args.exact
    lr = args.lr
    epochs = args.epochs
    d = args.d
    M_t = args.simulations

    P = random_simplex((size, size, 4))
    R = random_simplex((size, 4))
    mu = random_simplex((size,))
    
    #env = Tabular(P, R, mu, gamma)
    #env = Trap(size, gamma)
    env = Deepsea(size, gamma)


    H = 2 * horizon

    infos = []
    
    # TRY WITH LOGISTIC LEARNING RATE

    a = lr
    #lr = lambda k: a / (1 + np.exp(-10 * (k - epochs / 3) / epochs))
    #ys = [lr(k) for k in range(epochs)]

    for _ in range(n_agents):
        print(f'Running {alg.__name__} with depth {d}')
        infos.append(run(d=d))

    for df in infos:
        df.suboptimality.plot()

    plt.yscale('log')
    plt.show()

    print('Finished, exiting...')