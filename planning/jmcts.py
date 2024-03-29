import time
import chex
import jax
from jax import numpy as jnp
import mctx
from mctx import PolicyOutput

from functools import partial as par
from algorithms.policies import JPolicy

from gymnax.environments.environment import EnvState, EnvParams, Environment
from environments.jym.tabular import JTabular
from environments.solvable import *
from algorithms.policies import *


def rollout(env: JTabular, params: EnvParams, gamma: float,
            obs: chex.Array, state: EnvState, rng_key: chex.PRNGKey, pi: JPolicy
            ) -> float:
    def cond(a: tuple[float, chex.Array, EnvState, chex.PRNGKey, jnp.bool_, int]) -> jnp.bool_:
        reward, obs, state, key, done, t = a
        return ~done
    
    def step(a: tuple[float, chex.Array, EnvState, chex.PRNGKey, jnp.bool_, int]
             ) -> tuple[float, chex.Array, EnvState, chex.PRNGKey, jnp.bool_, int]:
        cumulative, obs, state, key, done, t = a
        ind = env.index(state)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, pi.act(ind))
        new_obs, new_state, reward, done, _ = env.step(subkey, state, action, params)
        cumulative += reward * gamma**t
        return cumulative, obs, new_state, key, done, t + 1
        
    reward, leaf_obs, leaf_state, key, done, t = jax.lax.while_loop(cond, step, (0, obs, state, rng_key, False, 0))
    
    return reward


def value_function(rollouts: int, val_pi: JPolicy, env: Environment, params: EnvParams, gamma: float,
                   obs: chex.Array, state: EnvState, rng_key: chex.PRNGKey) -> chex.Array:
    keys = jax.random.split(rng_key, rollouts)
    rolls = jax.vmap(rollout, (None, None, None, None, None, 0, None))(env, params, gamma, obs, state, keys, val_pi)
    return jnp.mean(rolls)


@par(jax.jit, static_argnums=(0, 2, 3, 4))
def root_fn(rollouts: int, val_pi: JPolicy, env: Environment, params: EnvParams,
            gamma: float, obs: chex.Array, state: EnvState, rng_key: chex.PRNGKey) -> mctx.RootFnOutput:
    return mctx.RootFnOutput(
        prior_logits = jnp.zeros((env.num_actions,)),
        value = value_function(rollouts, val_pi, env, params, gamma, obs, state, rng_key),
        embedding = (obs, state)
    )


@par(jax.jit, static_argnums=(0, 2, 3, 4))
def recurrent_fn(rollouts: int, val_pi: JPolicy, env: Environment, env_params: EnvParams, gamma: float,
                 params, rng_key: chex.PRNGKey,
                 action: chex.Array, embedding: tuple[chex.Array, EnvState]) -> mctx.RecurrentFnOutput:
    
    obs, state = embedding

    
    new_obs, new_state, reward, done, _ = env.step(rng_key, state, action, env_params)
    vf = value_function(rollouts, val_pi, env, env_params, gamma, obs, state, rng_key).astype(jnp.float32)
    value = jnp.where(done, 0, vf)
    
    recurrent_fn_output = mctx.RecurrentFnOutput(
        
        reward=reward,
        
        discount=jnp.where(done, 0, gamma),
        
        prior_logits=jnp.zeros((env.num_actions,)),
        
        value=value,
    )
    new_embedding = (new_obs, new_state)

    
    return recurrent_fn_output, new_embedding


@par(jax.jit, static_argnums=range(5))
def run_mcts(env: Environment,
             params: EnvParams,
             depth: int,
             num_simulations: int,
             rollouts: int,
             gamma: float,
             rng_key: chex.PRNGKey,
             obs: chex.Array,
             state: EnvState,
             val_pi: JPolicy) -> PolicyOutput:
    
    tree_key, value_key = jax.random.split(rng_key)

    batch_size = 1
    root_jit = jax.jit(par(root_fn, rollouts, val_pi, env, params, gamma))
    root = jax.vmap(root_jit, (None, None, 0))(obs, state, jax.random.split(value_key, batch_size))

    par_rfn_jit = jax.jit(par(recurrent_fn, rollouts, val_pi, env, params, gamma))
    rf = jax.vmap(par_rfn_jit, (None, None, 0, 0))  

    
    
    policy_output = mctx.muzero_policy(
        params=None,
        rng_key=tree_key,
        root=root,
        recurrent_fn=rf,
        num_simulations=num_simulations,
        max_depth=depth,
        qtransform=par(mctx.qtransform_by_min_max, min_value=-1, max_value=1),
        pb_c_init=1.21
    )
    return policy_output