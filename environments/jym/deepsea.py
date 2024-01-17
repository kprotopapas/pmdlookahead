from chex._src.pytypes import Array
from flax import struct
from typing import Callable

import chex
from jax import numpy as jnp
import jax
from gymnax.environments.environment import EnvState, Environment
import numpy as np
from functools import partial

from environments.jym.tabular import JTabular


@struct.dataclass
class EnvState:
    pos: chex.Array
    reward: chex.Array

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
        reward = 0
        state = EnvState(pos, reward)
        return self.get_obs(state), state


    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
             ) -> tuple[chex.Array, EnvState, float, bool, dict]:
        action = jnp.clip(action, 0, 1)
        position = state.pos
        done = position[0] >= self.size

        dx = 2 * action - 1
        delta = jnp.where(done, jnp.zeros((2, )), jnp.array([1, dx]))

        newpos = position + delta
        right_penalty = -0.1 * self.gamma**self.size / self.size

        reward = action * jnp.where(newpos[1] == self.size, 1, right_penalty)

        reward = jnp.where(done, 0, reward)
        
        done = newpos[0] >= self.size
        cumulative = state.reward + reward * self.gamma**position[0]
        new_state = EnvState(newpos, cumulative)

        obs = jnp.sum(jnp.array([self.size, 1]) * newpos).astype(jnp.int32)

        return obs, new_state, reward, done, {}
    

    def get_obs(self, state: EnvState) -> Array:
        return jnp.sum(self.embedding * state.pos).astype(jnp.int32)
    
    @property
    def embedding(self) -> chex.Array:
        return jnp.array([self.size, 1])
    
    def inverse(self, embedded: chex.Array) -> chex.Array:
        jnp.array([embedded // self.size, embedded % self.size])
