from planning.jmcts import *
from environments.jym import *
from environments.solvable import *


size = 32
depth = 3
gamma = 0.99
env = Trap(size, gamma)
jenv = JTrap(size=size, gamma=gamma)
embedding = lambda state: state



jpolicy = JSoftmaxPolicy(env.n_obs, env.n_act, embedding)
policy = SoftmaxPolicy(env.observation_space, env.action_space)

Q = env.Q(policy)
V = env.V(policy)
JV = jnp.array(V)

right, rright, _ = jenv.step(1)
left, rleft, _ = jenv.step(0)

key = jax.random.PRNGKey(42)


vright = value_function(right, key, jpolicy)
vleft = value_function(left, key, jpolicy)


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


lookahead = run_mcts(key, jenv, 10000, depth)
Q_hat = lookahead.search_tree.children_values[0, 0]
print('Real vs approx Q_d function')
print(Q_d[env.state])
print(Q_hat)

error = jnp.linalg.norm(Q_d[env.state] - Q_hat, ord=jnp.inf)
print(f'Error: {error}')
pass