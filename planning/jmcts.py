import chex
import jax
from jax import numpy as jnp
import mctx
from mctx import PolicyOutput

from functools import partial as par

from environments.jym import *
from environments.solvable import *
from algorithms.policies import *


def rollout(env: JEnv, rng_key: chex.PRNGKey, pi: JPolicy) -> Reward:
    reward_so_far = env.reward

    def cond(a: tuple[JEnv, chex.PRNGKey]) -> jnp.bool_:
        env, key = a
        return ~env.done
    
    def step(a: tuple[JEnv, chex.PRNGKey]) -> tuple[JEnv, chex.PRNGKey]:
        env, key = a
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, pi.act(env))
        env, reward, done = env.step(action)
        return env, key
    
    leaf, key = jax.lax.while_loop(cond, step, (env, rng_key))
    
    return leaf.reward - reward_so_far


def value_function(env: JEnv, rng_key: chex.PRNGKey, pi: JPolicy) -> chex.Array:
    #return JV[embedding(env.state)]

    keys = jax.random.split(rng_key, 1024)
    rolls = jax.vmap(rollout, (None, 0, None))(env, keys, pi)
    return jnp.mean(rolls)

    return rollout(env, rng_key).astype(jnp.float32)


def root_fn(env: JEnv, rng_key: chex.PRNGKey, pi: JPolicy) -> mctx.RootFnOutput:
    return mctx.RootFnOutput(
        prior_logits = jnp.zeros((2,)),
        value = value_function(env, rng_key, pi),
        embedding = env
    )


def recurrent_fn(params, rng_key: chex.PRNGKey, action: Action, embedding: JEnv) -> mctx.RecurrentFnOutput:
    # Extract the environment from the embedding.
    env = embedding
    discount = env.gamma

    # Play the action.
    env, reward, done = env.step(action)
    pi = params
    # Create the new MCTS node.
    recurrent_fn_output = mctx.RecurrentFnOutput(
        # reward for playing `action`
        reward=reward,
        # discount explained in the next section
        discount=discount,
        # prior for the new state
        prior_logits=jnp.zeros((2,)),
        # value for the new state
        value=jnp.where(done, 0, value_function(env, rng_key, pi)).astype(jnp.float32),
    )

    # Return the new node and the new environment.
    return recurrent_fn_output, env


@par(jax.jit, static_argnums=(3,4))
def run_mcts(rng_key: chex.PRNGKey, env: JEnv, depth: int, val_pi: JPolicy, num_simulations: int) -> PolicyOutput:
    batch_size = 1
    key1, key2 = jax.random.split(rng_key)

    
    root = jax.vmap(root_fn, (None, 0, None))(env, jax.random.split(key2, batch_size), val_pi)

    rf = jax.vmap(recurrent_fn, (None, None, 0, 0))
    policy_output = mctx.muzero_policy(
        params=val_pi,
        rng_key=key1,
        root=root,
        recurrent_fn=rf,
        num_simulations=num_simulations,
        max_depth=depth,
        dirichlet_fraction=0.0,
    )
    return policy_output