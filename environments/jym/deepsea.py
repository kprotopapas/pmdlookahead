from chex._src.pytypes import Array
from flax import struct
from typing import Callable

import chex
from jax import numpy as jnp
import jax
from gymnax.environments.environment import EnvState, Environment
from functools import partial

from environments.jym.tabular import JTabular


@struct.dataclass
class EnvState:
    pos: chex.Array

@struct.dataclass
class EnvParams:
    pass


class JDeepsea(JTabular):
    def __init__(self, size: int, gamma: float):
        self.penalty = -0.1 * gamma**size / size
        self.size = size
        self.goal = jnp.array([self.size, 0], dtype=jnp.int32)
        n_states = size * (size + 1) // 2 + 1

        def step(s, a):
            _, state = self.inverse(s)
            nstate = self.transition(state, a)
            sp = self.index(nstate)
            return jnp.zeros((n_states)).at[sp].set(1)


        states = jnp.arange(n_states)
        actions = jnp.arange(2)

        over_states = jax.vmap(step, (0, None), 1)
        over_both = jax.vmap(over_states, (None, 0), 2)
        P = over_both(states, actions)

        R = jnp.zeros((n_states, 2)).at[:, 1].set(self.penalty)
        R = R.at[n_states - 2, 1].set(1)
        R = R.at[n_states - 1, :].set(0)
        mu = jnp.zeros((n_states, )).at[0].set(1)

        super().__init__(P, R, mu, gamma, f'Deepsea')

    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params: EnvParams) -> tuple[chex.Array, EnvState]:
        pos = jnp.zeros((2, ), dtype=jnp.int32)
        state = EnvState(pos)
        return self.get_obs(state), state
    

    @partial(jax.jit, static_argnums=(0, ))
    def transition(self, state: EnvState, action: int) -> EnvState:
        position = state.pos

        dx = 2 * action - 1
        delta = jnp.array([1, dx], dtype=jnp.int32)

        new_pos = jnp.clip(position + delta, 0, self.size - 1)

        new_pos = jnp.where(position[0] >= self.size - 1, self.goal, new_pos)
        new_state = EnvState(new_pos)

        return new_state



    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
             ) -> tuple[chex.Array, EnvState, float, bool, dict]:
        action = jnp.clip(action, 0, 1)
        new_state = self.transition(state, action)
        new_pos = new_state.pos
        new_obs = self.get_obs(new_state)

        old_done = state.pos[0] >= self.size

        corner = jnp.allclose(state.pos, jnp.array([self.size - 1, self.size -1], dtype=jnp.int32))
        reward = action * jnp.where(corner, 1, self.penalty)
        reward = jnp.where(old_done, 0, reward)
        
        done = new_pos[0] >= self.size
        
        return new_obs, new_state, reward, done, {}
    

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: EnvState) -> Array:
        temp = jnp.zeros((self.size, self.size))
        row, column = state.pos
        return temp.at[row, column].set(1)

    
    @property
    def embedding(self) -> chex.Array:
        return jnp.array([self.size, 1])

    @partial(jax.jit, static_argnums=(0,))
    def index(self, state: EnvState) -> int:
        row, col = state.pos
        tri = row * (row + 1) // 2
        col = jax.lax.select(row >= self.size, 0, col)
        return tri + col

    
    @partial(jax.jit, static_argnums=(0,))
    def inverse(self, state_ind: int) -> tuple[chex.Array, EnvState]:
        row = jnp.floor((jnp.sqrt(8 * state_ind + 1) - 1) / 2).astype(jnp.int32) # returns closest triangle root
        tri = row * (row + 1) // 2
        col = jax.lax.select(row >= self.size, 0, state_ind - tri)

        pos = jnp.array([row, col])
        state = EnvState(pos)
        
        obs = self.get_obs(state)
        return obs, state
