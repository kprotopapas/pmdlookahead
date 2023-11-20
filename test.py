from environments.solvable import Trap, Deepsea
from algorithms.tree import BanditTS, MCTS
from algorithms.policies import SoftmaxPolicy
import numpy as np
from matplotlib import pyplot as plt
import wandb


size=32
env = Trap(size, 0.99)
pi = SoftmaxPolicy(env.observation_space, env.action_space)

state = size
depth = size // 2
tree = MCTS(env, pi, max_depth=depth, alpha=np.sqrt(2))

Q = env.Q(pi)
Q_opt = env.optimal()

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
n = 10000
for _ in range(n):
    tree.sample_trajectory(state=state)
    root = tree.roots[state].values
    error = np.max(np.abs(Q_d[state] - root))
    errors.append(error)


# ts = np.array([[tree.nodes[d][i].t for i in range(env.n_obs)] for d in range(depth + 1)])
# print(ts)

plt.plot(range(n), errors, 'r-')
#plt.plot(range(n), errors_2, 'b-')
plt.plot(range(n), np.ones((n, )) * sub, 'k--')

plt.show()