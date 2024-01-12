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
import gymnax

import wandb

args = parse_args()
group = args.group
depth = args.d
N = 50
M_l = 128


size = 16
depth = 8
gamma = 0.99
key = jax.random.PRNGKey(42)

env = Deepsea(size, gamma)
jenv = JDeepsea(size=size, gamma=gamma)
params = None

jenv, params = gymnax.make('DeepSea-bsuite', size=size)
obs, state = jenv.reset(key, params)

for i in range(12):
    obs, state, _, _, _ = jenv.step(key, state, 1, params)
    env.step(1)


embedding = jnp.arange(size**2, dtype=jnp.int32).reshape((size, size))
jpolicy = JSoftmaxPolicy.uniform(env.n_obs, env.n_act, embedding)
policy = SoftmaxPolicy(env.n_obs, env.n_act)

Q = env.Q(policy)
V = env.V(policy)
JV = jnp.array(V)

oright, sright, rright, _, _ = jenv.step(key, state, 1, params)
oleft, sleft, rleft, _, _ = jenv.step(key, state, 0, params)

key = jax.random.PRNGKey(42)


vf = jax.jit(value_function, static_argnums=(0, 2, 3))

vright = vf(M_l, jpolicy, jenv, params, gamma, oright, sright, key)
vleft = vf(M_l, jpolicy, jenv, params, gamma, oleft, sleft, key)


print('State')
print(state)
print()
print('Real vs approx Q function')
print(Q[env.state])
print(np.array([rleft + gamma * vleft, rright + gamma * vright]))
print()
root = jax.jit(root_fn, static_argnums=(0, 2, 3))
rt = root(M_l, jpolicy, jenv, params, gamma, obs, state, key)

recur = jax.jit(recurrent_fn, static_argnums=(0, 2, 3,))
precur = par(recur, M_l, jpolicy, jenv, params, gamma)
final = jax.jit(precur)
rfr = final(None, key, 1, (obs, state))
rfl = final(None, key, 0, (obs, state))
print('Root function')
print(rt)
print()

print('Recurrent Function Right')
print(rfr)
print()
print('Recurrent Function Left')
print(rfl)
print()

tdm = V
for _ in range(depth - 1):
    tdm = env.bellman(tdm)

gamma = env.gamma
next_V = np.sum(env.P * tdm[..., np.newaxis, np.newaxis], axis=0)
Q_d = env.R + gamma * next_V

start = time.time()

print(f'Running with {N} simulations')
lookahead: PolicyOutput = run_mcts(jenv, params, depth, N, M_l, gamma, key, obs, state, jpolicy)
Q_hat = gamma * lookahead.search_tree.children_values[0, 0] + lookahead.search_tree.children_rewards[0, 0]
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
lookahead2: PolicyOutput = run_mcts(jenv, params, depth, N, M_l, gamma, key, obs, state, jpolicy)
print(f'Took {time.time() - start} seconds')

print()
print('Running again')
start = time.time()
lookahead2: PolicyOutput = run_mcts(jenv, params, depth, N, M_l, gamma, key, obs, state, jpolicy)
print(f'Took {time.time() - start} seconds')

pass
# TODO SUBTREEEEEEEES