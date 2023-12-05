import mctx
import jax
from jax import numpy as jnp
from algorithms.policies import SoftmaxPolicy
import functools

from environments.jym import *


def rollout(env: Env, rng_key: chex.PRNGKey, policy: chex.Array) -> Reward:
    reward_so_far = env.reward

    def cond(a: tuple[Env, chex.PRNGKey]) -> jnp.bool_:
        env, key = a
        return ~env.done
    
    def step(a: tuple[Env, chex.PRNGKey]) -> tuple[Env, chex.PRNGKey]:
        env, key = a
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, policy(env))
        env, reward, done = env.step(action)
        return env, key
    
    leaf, key = jax.lax.while_loop(cond, step, (env, rng_key))
    
    return leaf.reward - reward_so_far


def value_function(env: Env, rng_key: chex.PRNGKey) -> chex.Array:
    return rollout(env, rng_key).astype(jnp.float32)


def root_fn(env: Env, rng_key: chex.PRNGKey) -> mctx.RootFnOutput:
    return mctx.RootFnOutput(
        prior_logits = jnp.zeros((2,)),
        value = value_function(env, rng_key),
        embedding = env
    )


def recurrent_fn(params, rng_key: chex.PRNGKey, action: Action, embedding: Env) -> mctx.RecurrentFnOutput:
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


@functools.partial(jax.jit, static_argnums=(2,))
def run_mcts(rng_key: chex.PRNGKey, env: Env, num_simulations: int) -> chex.Array:
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
        max_depth=env.size,
        dirichlet_fraction=0.0,
    )
    return policy_output



env = DeepseaX(size=10, gamma=0.99)
for _ in range(env.size - 3):
    env, _, _ = env.step(1)

print(env.state)
policy = run_mcts(jax.random.PRNGKey(0), env, 1000)
print(policy.action_weights)