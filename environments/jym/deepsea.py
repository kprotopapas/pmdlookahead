from flax import struct
from typing import Callable

import chex
from jax import numpy as jnp
import jax
from gymnax.environments.environment import Environment
import numpy as np
from functools import partial


@struct.dataclass
class EnvState:
    pos: chex.Array
    reward: chex.Array


@struct.dataclass
class EnvParams:
    pass


class JDeepsea(Environment):
    def __init__(self, size: int, gamma: float):
        super().__init__()
        self.size = size
        self.gamma = gamma


    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, key: chex.PRNGKey, params: EnvParams) -> tuple[chex.Array, EnvState]:
        pos = jnp.zeros((2, ), dtype=jnp.int32)
        reward = 0
        return pos, EnvState(pos, reward)


    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
             ) -> tuple[chex.Array, EnvState, float, bool, dict]:
        env = self
        action = jnp.clip(action, 0, 1)
        position = state.pos
        done = position[0] == self.size - 1
        dx = jnp.where(done, 0, action * 2 - 1)

        newpos = jnp.clip(jnp.array([position[0] + 1, position[1] + dx]), 0, env.size - 1)
        done = newpos[0] == env.size - 1
        right_penalty = -0.1 * env.gamma**env.size / env.size

        reward = action * jnp.where(newpos[1] == env.size - 1, 1, right_penalty)
        reward = jnp.where(done, 0, reward)
        cumulative = state.reward + reward * self.gamma**position[0]
        new_state = EnvState(newpos, cumulative)

        return newpos, new_state, reward, done, {}
    
    
    @property
    def embedding(self) -> chex.Array:
        return jnp.array([self.size, 1])
