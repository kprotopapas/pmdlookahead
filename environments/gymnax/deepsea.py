from chex._src.pytypes import Array
from flax import struct
from typing import Callable

import chex
from jax import numpy as jnp
import jax
from gymnax.environments.environment import EnvState, Environment
from functools import partial

from environments.gymnax.tabular import JTabular


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


        def step(s, a):
            _, state = self.inverse(s)
            nstate = self.transition(state, a)
            nobs = self.get_obs(nstate)
            sp = self.index(nobs)
            return jnp.zeros((size**2 + 1)).at[sp].set(1)


        states = jnp.arange(size**2 + 1)
        actions = jnp.arange(2)

        over_states = jax.vmap(step, (0, None), 1)
        over_both = jax.vmap(over_states, (None, 0), 2)
        P = over_both(states, actions)

        R = jnp.zeros((size**2 + 1, 2)).at[:, 1].set(self.penalty)
        R = R.at[size**2 - 1, 1].set(1)
        R = R.at[size**2, :].set(0)
        mu = jnp.zeros((size**2 + 1, )).at[0].set(1)

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
        ind = jnp.sum(jnp.array([self.size, 1]) * state.pos).astype(jnp.int32)
        return jnp.zeros((self.size**2 + 1, )).at[ind].set(1)

    
    @property
    def embedding(self) -> chex.Array:
        return jnp.array([self.size, 1])

    @partial(jax.jit, static_argnums=(0,))
    def index(self, obs: chex.Array) -> int:
        return jnp.where(jnp.any(obs > 0), jnp.argmax(obs), self.size**2)
    
    @partial(jax.jit, static_argnums=(0,))
    def inverse(self, state_index: int) -> tuple[chex.Array, EnvState]:
        row = state_index // self.size
        col = state_index % self.size

        pos = jnp.array([row, col])
        state = EnvState(pos)
        
        obs = self.get_obs(state)
        return obs, state
