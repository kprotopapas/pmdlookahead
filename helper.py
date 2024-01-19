import numpy as np
from jax import numpy as jnp
import jax
import argparse

MAX_FOR_EXP = 709


def simplex_project_sort(v: np.ndarray, z: float = 1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


@jax.jit
def jax_simplex_project_sort(v: jnp.ndarray):
    z = 1
    n_features = v.shape[0]
    u = jnp.sort(v)[::-1]
    cssv = jnp.cumsum(u) - z
    ind = jnp.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / jnp.float32(rho)
    w = jnp.maximum(v - theta, 0)
    return w


@jax.jit
def jax_simplex_project_bijection(v: jnp.ndarray):
    z = 1
    tau = 0.0001
    max_iter = 1000

    func = lambda x: jnp.sum(jnp.maximum(v - x, 0)) - z
    lower = jnp.min(v) - z / len(v)
    upper = jnp.max(v)
    midpoint = (upper + lower) / 2.0
    value = func(midpoint)

    for it in range(max_iter):
        if abs(value) <= tau:
            break

        if value <= 0:
            upper = midpoint
        else:
            lower = midpoint

    def cond(a: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
        lower, midpoint, upper, value, it = a
        return (jnp.abs(value) > tau) & (it < max_iter)

    def step(a: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]):
        lower, midpoint, upper, value, it = a
        midpoint = (upper + lower) / 2
        value = func(midpoint)

        upper = jnp.where(value <= 0, midpoint, upper)
        lower = jnp.where(value > 0, midpoint, lower)
        return lower, midpoint, upper, value, it + 1
    

    start = lower, midpoint, upper, value, 0
    lower, midpoint, upper, value, it = jax.lax.while_loop(cond, step, start)

    return np.maximum(v - midpoint, 0)



def random_simplex(shape: tuple):
    new_shape = (shape[0] + 1,) + shape[1:]
    temp = np.random.random(size=new_shape).astype(np.float64)
    temp[0, ...] = 0
    temp[1, ...] = 1
    sorted = np.sort(temp, axis=0)
    distr = sorted[1:, ...] - sorted[:-1, ...]
    return distr.astype(np.float64)


def logsumexp(t, axis=1):
    return np.log(np.sum(np.exp(t), axis=axis))


def parse_args(file=False):
    parser = argparse.ArgumentParser(prog='PMD TS runner', description='Runner for PMD TS', epilog='')
    if file:
        parser.add_argument('file', type=str)
    parser.add_argument('--env_name', type=str, default='Gridworld')
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--horizon', '-t', type=int, default=25)
    parser.add_argument('--gamma', '-g', type=float)

    parser.add_argument('--n_agents', '-n', type=int, default=5)
    parser.add_argument('--alg', '-a', type=str, default='DPQA')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', '-l', type=float, default=1.)
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('-d', type=int, default=1)
    parser.add_argument('--simulations', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--group', type=str)


    args = parser.parse_args()
    
    return args