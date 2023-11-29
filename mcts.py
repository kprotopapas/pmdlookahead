from environments.solvable import *
from algorithms.tree import BanditTS, MCTS
from algorithms.policies import SoftmaxPolicy
import numpy as np
from matplotlib import pyplot as plt
import wandb
from helper import parse_args
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    args = parse_args()
    group = args.group

    size = args.size
    alg = BanditTS if args.alg.upper() == 'BANDITTS' else MCTS
    use_wandb = args.wandb
    alpha = args.lr
    max_depth = args.d or size
    epochs = args.epochs
    horizon = args.horizon

    gamma = args.gamma or 1 - 1 / horizon

    env = Deepsea(size, gamma)
    pi = SoftmaxPolicy(env.observation_space, env.action_space)

    state = 0
    tree = BanditTS(env, pi, max_depth=max_depth, alpha=alpha, lam=0)


    tdm = env.V(pi)
    for i in range(max_depth - 1):
        tdm = env.bellman(tdm)

    next_V = np.sum(env.P * tdm[..., np.newaxis, np.newaxis], axis=0)
    Q_d = env.R + env.gamma * next_V

    best = np.max(Q_d[state])
    temp = np.copy(Q_d[state])

    i = np.argmax(Q_d[state])
    temp[i] = -np.inf

    sub = best - temp.max()


    errors = []
    rtgs = []
    old_error = np.inf
    wandb.init(project="pmd-ts", config={'algorithm': alg, 'env': env.name,
                                        'size': size, 'depth': max_depth, 'gamma': gamma,
                                        'alpha': alpha},
                                        group=group)
    for k in range(epochs):
        vmax, rtg = tree.sample_trajectory(state=state)
        root = tree.nodes[state].values
        qd = Q_d[state]
        error = np.max(np.abs(qd - root))
        errors.append(error)
        
        log = {'err': error, 'rtg': rtg}
        wandb.log(log)
        logging.info(f'Epoch: {k}, {log}')



    # ts = np.array([[tree.nodes[d][i].t for i in range(env.n_obs)] for d in range(depth + 1)])
    # print(ts)

    plt.plot(range(epochs), errors, 'r-')
    plt.plot(range(epochs), np.ones((epochs, )) * sub, 'k--')
    plt.yscale('log')

    plt.show()
    pass
