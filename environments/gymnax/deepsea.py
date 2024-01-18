from chex._src.pytypes import Array
from flax import struct
from typing import Callable

import chex
from jax import numpy as jnp
import jax
from gymnax.environments.environment import EnvState, Environment
import numpy as np
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
        super().__init__(size**2, 2, gamma)
        self.size = size
        self.gamma = gamma

    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params: EnvParams) -> tuple[chex.Array, EnvState]:
        pos = jnp.zeros((2, ), dtype=jnp.int32)
        state = EnvState(pos)
        return self.get_obs(state), state


    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
             ) -> tuple[chex.Array, EnvState, float, bool, dict]:
        action = jnp.clip(action, 0, 1)
        position = state.pos
        done = position[0] >= self.size

        dx = 2 * action - 1
        delta = jnp.where(done, jnp.zeros((2, ), dtype=jnp.int32), jnp.array([1, dx], dtype=jnp.int32))

        newpos = jnp.clip(position + delta, 0, self.size)
        new_state = EnvState(newpos)

        right_penalty = -0.1 * self.gamma**self.size / self.size # TODO different penalty to bsuite
        goal = (newpos[1] == self.size) & (newpos[0] == self.size)
        reward = action * jnp.where(goal, 1, right_penalty)
        reward = jnp.where(done, 0, reward)
        
        done = newpos[0] >= self.size

        new_obs = self.get_obs(new_state)

        return new_obs, new_state, reward, done, {}
    

    def get_obs(self, state: EnvState) -> Array:
        ind = jnp.sum(jnp.array([self.size, 1]) * state.pos).astype(jnp.int32)
        return jnp.zeros((self.n_states, )).at[ind].set(1)
    
    @property
    def embedding(self) -> chex.Array:
        return jnp.array([self.size, 1])
    
    @partial(jax.jit, static_argnums=(0,))
    def inverse(self, state_index: int) -> tuple[chex.Array, EnvState]:
        obs = jnp.zeros((self.n_states, )).at[state_index].set(1)
        row = state_index // self.size
        col = state_index % self.size
        pos = jnp.array([row, col])
        return obs, EnvState(pos)

    @partial(jax.jit, static_argnums=(0,))
    def index(self, obs: chex.Array) -> int:
        return jnp.argmax(obs)