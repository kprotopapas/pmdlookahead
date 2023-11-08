import numpy as np
from matplotlib import pyplot as plt

from environments.discrete import DiscreteStochastic, Trap
from baselines.algorithms import DEQA
from helper import random_simplex

if __name__ == '__main__':
    n_agents = 1
    size = 8
    horizon = 10
    alg = DEQA
    gamma = 1. - 1. / horizon
    epochs = 50

    n_actions = 8

    P = random_simplex((size, size, n_actions))
    R = random_simplex((size, n_actions))
    mu = random_simplex((size,))
    env = DiscreteStochastic(P, R, mu, gamma)
    env = Trap(size, 1, gamma)

    anything = 100

    deqa = DEQA(env, d=1)
    lr = lambda k: 0.1 / gamma**(anything * k) # TODO SHOWCASE
    lr = 1
    info = deqa.learn(epochs=epochs, use_wandb=False, exact=True, lr=lr)
    subs_base = [i['suboptimality'] for i in info]

    all_subs = []
    for d in range(5, 25, 5):
        deqa = DEQA(env, d=d)
        info = deqa.learn(epochs=epochs, use_wandb=False, exact=True, lr=lr)
        subs = [i['suboptimality'] for i in info]
        all_subs.append(subs)

    k = np.arange(epochs)

    plt.yscale('log')
    plt.plot(k, subs_base, 'r-')
    
    colors = ['b', 'c', 'y', 'm', 'g']

    for c, s in zip(colors, all_subs):
        plt.plot(k, s, f'{c}-')

    plt.plot(k, gamma**(2 * np.arange(epochs)) * subs_base[0], f'r--')

    for c, l in zip(colors, range(5, 25, 5)):
        plt.plot(k, gamma**(2 * l * np.arange(epochs)) * subs_base[0], f'{c}--')

    plt.ylim(1e-14, 1e2)

    plt.grid(True)
    plt.show()