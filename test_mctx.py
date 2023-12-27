import jax
from jax import numpy as jnp
import numpy as np
import time
from algorithms.policies import JSoftmaxPolicy

from environments.jym import *
from environments.solvable import *
from algorithms.policies import *
from planning.jmcts import *
from helper import *

import wandb

args = parse_args()
group = args.group
depth = args.d
N = 50
M_l = 128


size = 16
depth = 7
gamma = 0.99
env = Deepsea(size, gamma)
jenv = JDeepsea(size=size, gamma=gamma)

for i in range(size - depth): # 4 steps
    jenv, _, _ = jenv.step(1)
    env.step(1)


jpolicy = JSoftmaxPolicy.uniform(env.n_obs, env.n_act, jenv.embedding)
policy = SoftmaxPolicy(env.observation_space.n, env.action_space.n)

Q = env.Q(policy)
V = env.V(policy)
JV = jnp.array(V)

right, rright, _ = jenv.step(1)
left, rleft, _ = jenv.step(0)

key = jax.random.PRNGKey(42)

vright = value_function(M_l, jpolicy, right, key)
vleft = value_function(M_l, jpolicy, left, key)


print('State')
print(jenv.state)

print('Real vs approx Q function')
print(Q[env.state])
print(np.array([rleft + gamma * vleft, rright + gamma * vright]))

tdm = V
for _ in range(depth - 1):
    tdm = env.bellman(tdm)

gamma = env.gamma
next_V = np.sum(env.P * tdm[..., np.newaxis, np.newaxis], axis=0)
Q_d = env.R + gamma * next_V

start = time.time()

print(f'Running with {N} simulations')
lookahead: PolicyOutput = run_mcts(key, jenv, jpolicy, depth, N, M_l)
Q_hat = lookahead.search_tree.children_values[0, 0]
print(f'Total ran in {time.time() - start} seconds')
print()
print('Real vs approx Q_d function')
print(Q_d[env.state])
print(Q_hat)
error = np.linalg.norm(Q_d[env.state] - Q_hat, ord=np.inf)
print(f'Error: {error / np.max(np.abs(Q_d[env.state]))}')
print()

print('Running again')
start = time.time()
lookahead2: PolicyOutput = run_mcts(key, jenv, jpolicy, depth, N, M_l)
print(f'Took {time.time() - start} seconds')

print()
print('Running again')
start = time.time()
lookahead2: PolicyOutput = run_mcts(key, jenv, jpolicy, depth, N, M_l)
print(f'Took {time.time() - start} seconds')
pass

# TODO SUBTREEEEEEEES