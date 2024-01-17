from __future__ import annotations
from flax import struct
import chex
from jax import numpy as jnp
import jax
from functools import partial

from gymnax.environments.environment import Environment

from environments.gymnax.tabular import JTabular

@struct.dataclass
class EnvState:
    position: int
    reward: float
    time: int



@struct.dataclass
class EnvParams:
    pass


class JTrap(JTabular):
    def __init__(self, size: int, gamma: float):
        super().__init__(2 * size + 1, 2, gamma)
        self.size = size
        self.gamma = gamma
        self.reward_matrix: chex.Array = jnp.hstack([
            jnp.array([1], dtype=jnp.float32),
            jnp.ones((self.size - 1, ), dtype=jnp.float32) * self.gamma**self.size / self.size
        ])

        self.length = 2 * self.size + 1

    
    def reset(self, key: chex.PRNGKey, params: EnvParams) -> tuple[chex.Array, EnvState]:
        return 0, EnvState(0, 0, 0)


    @partial(jax.jit, static_argnums=(0))
    def step(self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
             ) -> tuple[chex.Array, EnvState, float, bool, dict]:
        position = state.position
        action = jnp.clip(action, 0, 1)
        done = state.position == self.size
        dx = jnp.where(done, 0, action * 2 - 1)

        newpos = jnp.clip(position + dx, -self.size, self.size)
        done = position == 0 or position == self.length - 1 or done
        reward = jnp.where(done, 0, self.reward_matrix[newpos + self.size])
        cumulative = state.reward + self.gamma**state.time * reward
        
        new_state = EnvState(newpos, cumulative, state.time + 1)


        return newpos, new_state, reward, done, {}