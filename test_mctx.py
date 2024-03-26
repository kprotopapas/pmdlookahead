import jax
from jax import numpy as jnp
import numpy as np
import time
from algorithms.policies import JSoftmaxPolicy
from algorithms.tree import PGMCTS

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
N = args.simulations
M_l = 128


size = args.size
gamma = 0.99
key = jax.random.PRNGKey(42)

jenv = DeepSea(size, gamma)
params = jenv.params

obs, state = jenv.reset(key, params)

for i in range(12):
    obs, state, _, _, _ = jenv.step(key, state, 1, params)


jpolicy = JSoftmaxPolicy.uniform(jenv.num_states, jenv.num_actions)

Q = jenv.Q(jpolicy)
V = jenv.V(jpolicy)
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
s = jenv.index(state)
print(Q[s])
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
    tdm = jenv.bellman(tdm)

gamma = jenv.gamma
next_V = np.sum(jenv.P * tdm[..., np.newaxis, np.newaxis], axis=0)
Q_d = jenv.R + gamma * next_V

start = time.time()

print(f'Running with {N} simulations')
lookahead: PolicyOutput = run_mcts(jenv, params, depth, N, M_l, gamma, key, obs, state, jpolicy)
Q_hat = gamma * lookahead.search_tree.children_values[0, 0] + lookahead.search_tree.children_rewards[0, 0]
print(f'Total ran in {time.time() - start} seconds')
print()
print('Real vs approx Q_d function')
print(Q_d[s])
print(Q_hat)
error = np.linalg.norm(Q_d[s] - Q_hat, ord=np.inf)
print(f'Error: {error / np.max(np.abs(Q_d[s]))}')
print()

print('Running again')
start = time.time()
lookahead2: PolicyOutput = run_mcts(jenv, params, depth, N, M_l, gamma, key, obs, state, jpolicy)
print(f'Took {time.time() - start} seconds')

print()
print('Running again')
start = time.time()
lookahead2: PolicyOutput = run_mcts(jenv, params, depth, N, M_l, gamma, key, obs, state, jpolicy)
duration = time.time() - start
print(f'Took {duration} seconds')

alg = PGMCTS(jenv, depth=depth, M_l=M_l, M_t = N, horizon=size)
print(f'Without vectorisation would take {duration * alg.horizon} to estimate value')

print()

print(f'Sampling trajectory')
indices, actions, rewards = alg.sample_trajectory(key, jenv, params, alg.pi)
start = time.time()
traj_key, tree_key = jax.random.split(key)
indices, actions, rewards = alg.sample_trajectory(traj_key, jenv, params, alg.pi)
print(f'Took {time.time() - start} seconds')

i = indices[0]
obs, state = jenv.inverse(i)
output: PolicyOutput = run_mcts(jenv, params, depth, N, M_l, gamma, key, obs, state, alg.pi)
tree = output.search_tree
grad_k = tree.children_rewards[0, 0] + jenv.gamma * tree.children_values[0, 0]

start = time.time()
obs, state = jenv.inverse(i)
output: PolicyOutput = run_mcts(jenv, params, depth, N, M_l, gamma, key, obs, state, alg.pi)
tree = output.search_tree
grad_k = tree.children_rewards[0, 0] + jenv.gamma * tree.children_values[0, 0]
end = time.time()

print(f'Should take {end - start} seconds, total {(end - start) * len(indices)} for {len(indices)} states')

alg = PGMCTS(jenv, size, depth, 1, M_l, args.simulations)


alg.sample_trajectory(key, jenv, params, alg.pi)

key, new_key = jax.random.split(key)
start = time.time()
traj = alg.sample_trajectory(new_key, jenv, params, alg.pi)
end = time.time()

print(f'Sample trajectory in {end - start} seconds')

alg.estimate_value(key, jenv, params, alg.pi)

key, new_key = jax.random.split(key)
start = time.time()
val = alg.estimate_value(new_key, jenv, params, alg.pi)
end = time.time()

print(f'Estimate value in {end - start} seconds')

indices, _, _ = traj
print(len(indices))