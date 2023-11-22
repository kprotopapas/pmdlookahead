from environments.solvable import Trap, Deepsea
from algorithms.tree import BanditTS, MCTS
from algorithms.policies import SoftmaxPolicy
import numpy as np
from matplotlib import pyplot as plt
import wandb
from helper import parse_args

if __name__ == '__main__':
    args = parse_args()
    group = args.group

    size = args.size
    alg = BanditTS if args.alg.upper() == 'BANDITTS' else MCTS
    use_wandb = args.wandb
    alpha = args.lr
    max_depth = args.d or size
    n = args.epochs
    horizon = args.horizon

    gamma = args.gamma or 1 - 1 / horizon

    env = Deepsea(size, gamma)
    pi = SoftmaxPolicy(env.observation_space, env.action_space)

    state = size**2 - max_depth * size - max_depth
    tree = MCTS(env, pi, max_depth=max_depth, alpha=alpha)


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
    for i in range(n):
        vmax, rtg = tree.sample_trajectory(state=state)
        root = tree.roots[state].values
        #root = tree.nodes[0][state].values
        qd = Q_d[state]
        error = np.max(np.abs(qd - root)) / np.max(qd)
        errors.append(error)
        if rtg / i > 1e-2:
            pass
        if error - old_error > 1e-3:
            pass
        if error < 1e-2:
            pass
        old_error = error
        old_t = tree.roots[state].t
        old_s = np.copy(tree.roots[state].s)
        rtgs.append(rtg)

        wandb.log({'err': error, 'rtg': rtg})



    # ts = np.array([[tree.nodes[d][i].t for i in range(env.n_obs)] for d in range(depth + 1)])
    # print(ts)

    plt.plot(range(n), errors, 'r-')
    plt.plot(range(n), np.ones((n, )) * sub, 'k--')
    plt.yscale('log')

    plt.show()
    pass
