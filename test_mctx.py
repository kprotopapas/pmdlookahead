import jax
from jax import numpy as jnp
import time

from environments.jym import *
from environments.solvable import *
from algorithms.policies import *
from planning.jmcts import *


size = 32
depth = 7
gamma = 0.99
env = Deepsea(size, gamma)
jenv = JDeepsea(size=size, gamma=gamma)

for i in range(size - depth - 3):
    jenv, _, _ = jenv.step(1)
    env.step(1)


embedding = lambda state: state[0] * size + state[1]
jpolicy = JSoftmaxPolicy(env.n_obs, env.n_act, embedding)
policy = SoftmaxPolicy(env.observation_space, env.action_space)

Q = env.Q(policy)
V = env.V(policy)
JV = jnp.array(V)

right, rright, _ = jenv.step(1)
left, rleft, _ = jenv.step(0)

key = jax.random.PRNGKey(42)

vright = value_function(128, right, key, jpolicy)
vleft = value_function(128, left, key, jpolicy)


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
N = 1000
lookahead: PolicyOutput = run_mcts(key, jenv, depth, jpolicy, N, 128)
Q_hat = lookahead.search_tree.children_values[0, 0]
print('Real vs approx Q_d function')
print(Q_d[env.state])
print(Q_hat)

error = jnp.linalg.norm(Q_d[env.state] - Q_hat, ord=jnp.inf)
print(f'Error: {error / np.max(np.abs(Q_d[env.state]))}')

print(f'Took {time.time() - start} seconds')


# TODO SUBTREEEEEEEES