from typing import Optional
import mctx
import jax
from jax import numpy as jnp
from algorithms.policies import SoftmaxPolicy
from functools import partial as par

from environments.jym import *
from environments.solvable import *
from mctx._src import action_selection
from mctx._src import base
from mctx._src import qtransforms
from mctx._src import search
from mctx._src.search import simulate, Tree, expand, backward
from mctx._src.policies import _apply_temperature, _get_logits_from_probs


class MCTS(object):
    pass


def rollout(env: JEnv, rng_key: chex.PRNGKey, policy: JPolicy) -> Reward:
    reward_so_far = env.reward

    def cond(a: tuple[JEnv, chex.PRNGKey]) -> jnp.bool_:
        env, key = a
        return ~env.done
    
    def step(a: tuple[JEnv, chex.PRNGKey]) -> tuple[JEnv, chex.PRNGKey]:
        env, key = a
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, policy.act(env))
        env, reward, done = env.step(action)
        return env, key
    
    leaf, key = jax.lax.while_loop(cond, step, (env, rng_key))
    
    return leaf.reward - reward_so_far


def value_function(env: JEnv, rng_key: chex.PRNGKey, policy: JPolicy = None) -> chex.Array:
    if policy is None:
        policy = JSoftmaxPolicy(env)
    return rollout(env, rng_key, policy).astype(jnp.float32)


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


@par(jax.jit, static_argnums=(2,))
def run_mcts(rng_key: chex.PRNGKey, env: JEnv, num_simulations: int, depth: int) -> chex.Array:
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
        max_depth=2,
        dirichlet_fraction=0.0,
    )
    return policy_output



def cont_mcts(
    policy_output: base.PolicyOutput[None],
    params: base.Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
    dirichlet_fraction: chex.Numeric = 0.25,
    dirichlet_alpha: chex.Numeric = 0.3,
    pb_c_init: chex.Numeric = 1.25,
    pb_c_base: chex.Numeric = 19652,
    temperature: chex.Numeric = 1.0,
    start: chex.Numeric = 0) -> base.PolicyOutput[None]:


    # Running the search.
    interior_action_selection_fn = par(
        action_selection.muzero_action_selection,
        pb_c_base=pb_c_base,
        pb_c_init=pb_c_init,
        qtransform=qtransform)
    
    root_action_selection_fn = par(
        interior_action_selection_fn,
        depth=0)

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

    # Allocate all necessary storage.
    search_tree = policy_output.search_tree
    _, search_tree = loop_fn(
        start, start + num_simulations, body_fun, (rng_key, search_tree))
        

    # Sampling the proposed action proportionally to the visit counts.
    summary = search_tree.summary()
    action_weights = summary.visit_probs
    action_logits = _apply_temperature(
        _get_logits_from_probs(action_weights), temperature)
    action = jax.random.categorical(rng_key, action_logits)
    return base.PolicyOutput(
        action=action,
        action_weights=action_weights,
        search_tree=search_tree)
