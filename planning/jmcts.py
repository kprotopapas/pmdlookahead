import time
import chex
import jax
from jax import numpy as jnp
import mctx
from mctx import PolicyOutput

from typing import Optional, Any

from mctx._src.search import Tree, expand, backward, simulate ,instantiate_tree_from_root
from mctx._src.policies import _apply_temperature, _get_logits_from_probs, _add_dirichlet_noise, _mask_invalid_actions
from mctx._src import action_selection
from mctx._src import base
from mctx._src import qtransforms
from mctx._src import search
from mctx._src import seq_halving

from functools import partial as par
from algorithms.policies import JPolicy

from environments.jym import *
from environments.solvable import *
from algorithms.policies import *


@jax.jit
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


@par(jax.jit, static_argnums=(0,))
def value_function(rollouts: int, val_pi: JPolicy, env: JEnv, rng_key: chex.PRNGKey) -> chex.Array:
    keys = jax.random.split(rng_key, rollouts)
    rolls = jax.vmap(rollout, (None, 0, None))(env, keys, val_pi)
    return jnp.mean(rolls)


@par(jax.jit, static_argnums=(0,))
def root_fn(rollouts: int, val_pi: JPolicy, env: JEnv, rng_key: chex.PRNGKey) -> mctx.RootFnOutput:
    return mctx.RootFnOutput(
        prior_logits = jnp.zeros((2,)),
        value = value_function(rollouts, val_pi, env, rng_key),
        embedding = env
    )


@par(jax.jit, static_argnums=(0,))
def recurrent_fn(rollouts: int, val_pi: JPolicy, params, 
                 rng_key: chex.PRNGKey, action: Action, embedding: JEnv) -> mctx.RecurrentFnOutput:
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
        value=value_function(rollouts, val_pi, env, rng_key).astype(jnp.float32),
    )

    # Return the new node and the new environment.
    return recurrent_fn_output, env


@par(jax.jit, static_argnums=(3, 4, 5))
def run_mcts(rng_key: chex.PRNGKey,
             env: JEnv,
             val_pi: JPolicy,
             depth: int,
             num_simulations: int,
             rollouts: int) -> PolicyOutput:
    
    key1, key2 = jax.random.split(rng_key)

    batch_size = 1
    root_jit = jax.jit(par(root_fn, rollouts, val_pi))
    root = jax.vmap(root_jit, (None, 0))(env, jax.random.split(key2, batch_size))

    rfn_jit = jax.jit(par(recurrent_fn, rollouts, val_pi))
    rf = jax.vmap(rfn_jit, (None, None, 0, 0))  

    #policy = jax.jit(mctx.muzero_policy, static_argnames=('temperature', 'recurrent_fn', 'num_simulations', 'loop_fn', 'qtransform', 'max_depth', 'dirichlet_fraction', 'dirichlet_alpha', 'pb_c_init', 'pb_c_base'))
    
    policy_output = mctx.muzero_policy(
        params=None,
        rng_key=key1,
        root=root,
        recurrent_fn=rf,
        num_simulations=num_simulations,
        max_depth=depth,
        qtransform=par(mctx.qtransform_by_min_max, min_value=-1, max_value=1),
        pb_c_init=1.21
    )
    return policy_output