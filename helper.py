import numpy as np
import argparse

MAX_FOR_EXP = 709


def simplex_project_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def random_simplex(shape):
    new_shape = (shape[0] + 1,) + shape[1:]
    temp = np.random.random(size=new_shape).astype(np.float64)
    temp[0, ...] = 0
    temp[1, ...] = 1
    sorted = np.sort(temp, axis=0)
    distr = sorted[1:, ...] - sorted[:-1, ...]
    return distr.astype(np.float64)


def logsumexp(t, axis=1):
    return np.log(np.sum(np.exp(t), axis=axis))


def parse_args():
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
    parser.add_argument('-d', type=int, default=1)
    parser.add_argument('--simulations', type=int)

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--group', type=str)


    args = parser.parse_args()
    
    return args