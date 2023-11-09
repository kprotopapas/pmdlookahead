import numpy as np

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