from environments.solvable import Trap, Deepsea
from algorithms.tree import BanditTS, MCTS
from algorithms.policies import SoftmaxPolicy
import numpy as np
from matplotlib import pyplot as plt
import wandb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--group', type=str, default='mcts_v1')

args = parser.parse_args()
group = args.group

size=13
gamma = 0.9
env = Trap(size, gamma)
pi = SoftmaxPolicy(env.observation_space, env.action_space)

state = size
depth = size
tree = MCTS(env, pi, max_depth=depth, alpha=np.sqrt(2))


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
n = 1000
wandb.init(project="pmd-ts", config={'algorithm': 'MCTS', 'env': 'Deepsea',
                                     'size': size, 'depth': depth, 'gamma': gamma},
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