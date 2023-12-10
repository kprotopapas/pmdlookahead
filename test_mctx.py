import chex
import jax
from jax import numpy as jnp
import mctx

from functools import partial as par

from environments.jym import *
from environments.solvable import *
from algorithms.policies import *


def rollout(env: JEnv, rng_key: chex.PRNGKey) -> Reward:
    reward_so_far = env.reward

    def cond(a: tuple[JEnv, chex.PRNGKey]) -> jnp.bool_:
        env, key = a
        return ~env.done
    
    def step(a: tuple[JEnv, chex.PRNGKey]) -> tuple[JEnv, chex.PRNGKey]:
        env, key = a
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, jpolicy.act(env))
        env, reward, done = env.step(action)
        return env, key
    
    leaf, key = jax.lax.while_loop(cond, step, (env, rng_key))
    
    return leaf.reward - reward_so_far


def value_function(env: JEnv, rng_key: chex.PRNGKey) -> chex.Array:
    return JV[embedding(env.state)]
    return rollout(env, rng_key).astype(jnp.float32)


def root_fn(env: JEnv, rng_key: chex.PRNGKey) -> mctx.RootFnOutput:
    return mctx.RootFnOutput(
        prior_logits = jnp.zeros((2,)),
        value = value_function(env, rng_key),
        embedding = env
    )


def recurrent_fn(params, rng_key: chex.PRNGKey, action: Action, embedding: JEnv) -> mctx.RecurrentFnOutput:
    # Extract the environment from the embedding.
    env = embedding
    discount = env.gamma

    # Play the action.
    env, reward, done = env.step(action)

    # Create the new MCTS node.
    recurrent_fn_output = mctx.RecurrentFnOutput(
        # reward for playing `action`
        reward=reward,
        # discount explained in the next section
        discount=discount,
        # prior for the new state
        prior_logits=jnp.zeros((2,)),
        # value for the new state
        value=jnp.where(done, 0, value_function(env, rng_key)).astype(jnp.float32),
    )

    # Return the new node and the new environment.
    return recurrent_fn_output, env


@par(jax.jit, static_argnums=(3,))
def run_mcts(rng_key: chex.PRNGKey, env: JEnv, depth: int, num_simulations: int) -> chex.Array:
    batch_size = 1
    key1, key2 = jax.random.split(rng_key)
    
    root = jax.vmap(root_fn, (None, 0))(env, jax.random.split(key2, batch_size))

    rf = jax.vmap(recurrent_fn, (None, None, 0, 0))
    policy_output = mctx.muzero_policy(
        params=None,
        rng_key=key1,
        root=root,
        recurrent_fn=rf,
        num_simulations=num_simulations,
        max_depth=depth,
        dirichlet_fraction=0.0,
    )
    return policy_output


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

vright = value_function(right, key)
vleft = value_function(left, key)


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


lookahead = run_mcts(key, jenv, depth, 10000)
Q_hat = lookahead.search_tree.children_values[0, 0]
print('Real vs approx Q_d function')
print(Q_d[env.state])
print(Q_hat)

error = jnp.linalg.norm(Q_d[env.state] - Q_hat, ord=jnp.inf)
print(f'Error: {error}')
pass