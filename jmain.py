import numpy as np
import argparse
from matplotlib import pyplot as plt
import time
import pandas as pd
from algorithms.tree import PGMCTS

from environments.jym import *
from environments.jym.my_deepsea import JDeepsea
from algorithms.jym import *
from helper import random_simplex, parse_args

def run(d=1):
    if alg == PGMCTS:
        agent = alg(env, horizon=H, depth=d, M_t=M_t, M_l=M_l) # time per iteration 0.3 s
    elif alg in [JDPQA, JDEQA]:
        agent = alg(env, horizon=H, d=d, M_t=M_t, M_l=M_l, M_f=M_f)
    else:
        agent = alg(env, horizon=H, n_trajectories=M_f) # time per iteration ~10s
        
    start = time.time()
    info_dpmd = agent.learn(epochs, exact=exact, use_wandb=use_wandb,
                            verbose=True, lr=lr, group=group, seed=seed)
    time_dpmd = time.time() - start
    print(f'{alg.__name__} with depth {d} took time {time_dpmd}')

    info_df = pd.DataFrame.from_records(info_dpmd)
    info_df.set_index('epoch', drop=True)
    return info_df


def parse_alg(alg: str):
    match alg.upper():
        case 'REINFORCE':
            return JREINFORCE
        case 'DPQA':
            return JDPQA
        case 'DEQA':
            return JDEQA
        case 'PQA':
            return JPQA
        case 'EQA':
            return JEQA
        case 'PGMCTS':
            return PGMCTS
        case 'PI':
            return JPolicyIteration


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
    seed = args.seed

    P = random_simplex((size, size, 4))
    R = random_simplex((size, 4))
    mu = random_simplex((size,))
    
    #env = JTabular(P, R, mu, gamma, 'Stochastic')
    #env = Trap(size, gamma)
    env = DeepSea(size, gamma, deterministic=False)
    #env = JDeepsea(size, gamma)
    #env = Catch(gamma, rows=13, columns=13)

    #env = Gridworld(gamma, rows=48, columns=48, deterministic=True, seed=seed)

    # Formula from (Johnson 2023)
    epsilon = 1e-4
    delta = 0.01
    K = np.log(4 / ((1 - gamma) * epsilon)) / (1 - gamma)
    H = np.log(16 / (epsilon * np.power(1 - gamma, 3))) / (1 - gamma)
    M_f = np.power(gamma, - 2 * H) * np.log(2 * K * env.num_states * env.num_actions / delta) / 2

    K, H, M_f = int(np.ceil(K)), int(np.ceil(H)), int(np.ceil(M_f))

    print(f'K: {K}')
    print(f'H: {H}')
    print(f'M_f: {M_f}')


    H = size # fixed for deepsea because larger doesn't matter
    M_f = 16 # deterministic
    M_t = 16 # deterministic
    M_l = 16 # Winnicki says we can do one episode

    infos = []
    
    # TRY WITH LOGISTIC LEARNING RATE

    a = lr
    #lr = lambda k: a / (1 + np.exp(-10 * (k - epochs / 3) / epochs))
    #lr = lambda k: 0.1
    #ys = [lr(k) for k in range(epochs)]

    for _ in range(n_agents):
        print(f'Running {alg.__name__} with depth {d}')
        infos.append(run(d=d))

    for df in infos:
        df.suboptimality.plot()

    #plt.yscale('log')
    plt.show()

    print('Finished, exiting...')