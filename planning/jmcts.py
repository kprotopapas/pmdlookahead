import chex
import jax
from jax import numpy as jnp
import mctx
from mctx import PolicyOutput

from typing import Optional, Any

from mctx._src.search import Tree, expand, backward, simulate
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


def value_function(rollouts: int, env: JEnv, rng_key: chex.PRNGKey, pi: JPolicy) -> chex.Array:
    #return JV[embedding(env.state)]

    keys = jax.random.split(rng_key, rollouts)
    rolls = jax.vmap(rollout, (None, 0, None))(env, keys, pi)
    return jnp.mean(rolls)


def root_fn(rollouts: int, env: JEnv, rng_key: chex.PRNGKey, pi: JPolicy) -> mctx.RootFnOutput:
    return mctx.RootFnOutput(
        prior_logits = jnp.zeros((2,)),
        value = value_function(rollouts, env, rng_key, pi),
        embedding = env
    )


def recurrent_fn(params, rng_key: chex.PRNGKey, action: Action, embedding: JEnv) -> mctx.RecurrentFnOutput:
    # Extract the environment from the embedding.
    env = embedding
    discount = env.gamma

    # Play the action.
    env, reward, done = env.step(action)
    pi = params['pi']
    rollouts = params['rollouts']
    # Create the new MCTS node.
    recurrent_fn_output = mctx.RecurrentFnOutput(
        # reward for playing `action`
        reward=reward,
        # discount explained in the next section
        discount=discount,
        # prior for the new state
        prior_logits=jnp.zeros((2,)),
        # value for the new state
        value=jnp.where(done, 0, value_function(rollouts, env, rng_key, pi)).astype(jnp.float32),
    )

    # Return the new node and the new environment.
    return recurrent_fn_output, env


@par(jax.jit, static_argnums=range(2, 6))
def run_mcts(rng_key: chex.PRNGKey,
             env: JEnv,
             depth: int,
             val_pi: JPolicy,
             num_simulations: int,
             num_rollouts: int) -> PolicyOutput:
    batch_size = 1
    key1, key2 = jax.random.split(rng_key)

    params = {'pi': val_pi, 'rollouts': num_rollouts}
    root = jax.vmap(root_fn, (None, None, 0, None))(num_rollouts, env, jax.random.split(key2, batch_size), val_pi)

    rf = jax.vmap(recurrent_fn, (None, None, 0, 0))
    policy_output = mctx.muzero_policy(
        params=params,
        rng_key=key1,
        root=root,
        recurrent_fn=rf,
        num_simulations=num_simulations,
        max_depth=depth,
        dirichlet_fraction=0.0,
    )
    return policy_output


def cont_search(
    start_tree: Tree,
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    start_idx: int = 0,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop) -> Tree:
  
    action_selection_fn = action_selection.switching_action_selection_wrapper(
        root_action_selection_fn=root_action_selection_fn,
        interior_action_selection_fn=interior_action_selection_fn
    )

    # Do simulation, expansion, and backward steps.
    batch_size = root.value.shape[0]
    batch_range = jnp.arange(batch_size)
    if max_depth is None:
        max_depth = num_simulations
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits)

    def body_fun(sim, loop_state):
        rng_key, tree = loop_state
        rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
        # simulate is vmapped and expects batched rng keys.
        simulate_keys = jax.random.split(simulate_key, batch_size)
        parent_index, action = simulate(
        simulate_keys, tree, action_selection_fn, max_depth)
        # A node first expanded on simulation `i`, will have node index `i`.
        # Node 0 corresponds to the root node.
        next_node_index = tree.children_index[batch_range, parent_index, action]
        next_node_index = jnp.where(next_node_index == Tree.UNVISITED,
                                sim + 1, next_node_index)
        tree = expand(
            params, expand_key, tree, recurrent_fn, parent_index,
            action, next_node_index)
        tree = backward(tree, next_node_index)
        loop_state = rng_key, tree
        return loop_state
    
    _, tree = loop_fn(
    start_idx, start_idx + num_simulations, body_fun, (rng_key, start_tree))

    return tree


def cont_mctx(tree: Tree,
              rng_key: chex.PRNGKey,
              env: JEnv,
              depth: int,
              val_pi: JPolicy,
              num_simulations: int,
              start_idx: int,
              qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
              dirichlet_fraction: chex.Numeric = 0.25,
              dirichlet_alpha: chex.Numeric = 0.3,
              pb_c_init: chex.Numeric = 1.25,
              pb_c_base: chex.Numeric = 19652,
              temperature: chex.Numeric = 1.) -> PolicyOutput:
    
    batch_size = 1
    key1, key2 = jax.random.split(rng_key)

    
    root = jax.vmap(root_fn, (None, 0, None))(env, jax.random.split(key2, batch_size), val_pi)
    rf = jax.vmap(recurrent_fn, (None, None, 0, 0))
    
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings
    pb_c_init: chex.Numeric = 1.25
    pb_c_base: chex.Numeric = 19652

    rng_key, dirichlet_rng_key, search_rng_key = jax.random.split(rng_key, 3)

    noisy_logits = _get_logits_from_probs(
      _add_dirichlet_noise(
          dirichlet_rng_key,
          jax.nn.softmax(root.prior_logits),
          dirichlet_fraction=dirichlet_fraction,
          dirichlet_alpha=dirichlet_alpha))
    invalid_actions = None
    root = root.replace(
      prior_logits=_mask_invalid_actions(noisy_logits, invalid_actions))


    interior_action_selection_fn = par(
        action_selection.muzero_action_selection,
        pb_c_base=pb_c_base,
        pb_c_init=pb_c_init,
        qtransform=qtransform)
    
    root_action_selection_fn = par(
        interior_action_selection_fn,
        depth=0)
    
    search_tree = cont_search(start_tree=tree, params=val_pi, rng_key=search_rng_key,
                       root=root,
                       recurrent_fn=rf,
                       root_action_selection_fn=root_action_selection_fn,
                       interior_action_selection_fn=interior_action_selection_fn,
                       num_simulations=num_simulations,
                       start_idx=start_idx,
                       max_depth=depth,
                       invalid_actions=invalid_actions,
                       loop_fn=jax.lax.fori_loop)
    
    summary = search_tree.summary()
    action_weights = summary.visit_probs
    action_logits = _apply_temperature(
        _get_logits_from_probs(action_weights), temperature)
    action = jax.random.categorical(rng_key, action_logits)
    return base.PolicyOutput(
        action=action,
        action_weights=action_weights,
        search_tree=search_tree)