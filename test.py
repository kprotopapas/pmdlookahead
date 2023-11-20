from environments.solvable import Trap, Deepsea
from algorithms.tree import BanditTS, MCTS
from algorithms.policies import SoftmaxPolicy
import numpy as np
from matplotlib import pyplot as plt
import wandb
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='mcts_v1')
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--horizon', '-t', type=int, default=25)
    parser.add_argument('--gamma', '-g', type=float)

    parser.add_argument('--alg', '-a', type=str, default='MCTS')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', '-l', type=float, default=np.sqrt(2))
    parser.add_argument('-d', type=int)

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--group', type=str)

    args = parser.parse_args()
    group = args.group

    size = args.size
    alg = BanditTS if args.alg.upper() == 'BANDITTS' else MCTS
    use_wandb = args.wandb
    alpha = args.lr
    depth = args.d or size
    n = args.epochs
    horizon = args.horizon

    gamma = args.gamma or 1 - 1 / horizon

    env = Trap(size, gamma)
    pi = SoftmaxPolicy(env.observation_space, env.action_space)

    state = size
    tree = MCTS(env, pi, max_depth=depth, alpha=alpha)


    tdm = env.V(pi)
    for i in range(depth - 1):
        tdm = env.bellman(tdm)

    next_V = np.sum(env.P * tdm[..., np.newaxis, np.newaxis], axis=0)
    Q_d = env.R + env.gamma * next_V

    best = np.max(Q_d[state])
    temp = np.copy(Q_d[state])

    i = np.argmax(Q_d[state])
    temp[i] = -np.inf

    sub = best - temp.max()


    errors = []
    wandb.init(project="pmd-ts", config={'algorithm': alg, 'env': 'Trap',
                                        'size': size, 'depth': depth, 'gamma': gamma,
                                        'alpha': alpha},
                                        group=group)
    for _ in range(n):
        tree.sample_trajectory(state=state)
        root = tree.roots[state].values
        #root = tree.nodes[0][state].values
        qdnorm = Q_d[state] - Q_d[state].max()
        rnorm = root - root.max()
        error = np.max(np.abs(qdnorm - rnorm))
        errors.append(error)
        wandb.log({'err': error})


    # ts = np.array([[tree.nodes[d][i].t for i in range(env.n_obs)] for d in range(depth + 1)])
    # print(ts)

    plt.plot(range(n), errors, 'r-')
    plt.plot(range(n), np.ones((n, )) * sub, 'k--')
    plt.yscale('log')

    plt.show()