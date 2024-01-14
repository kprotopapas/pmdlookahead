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
    pass


@struct.dataclass
class EnvParams:
    pass


class JTabular(Environment):
    def __init__(self, n_states: int, n_actions: int, gamma: float):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma


    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, key: chex.PRNGKey, params: EnvParams) -> tuple[chex.Array, EnvState]:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
             ) -> tuple[chex.Array, EnvState, float, bool, dict]:
        raise NotImplementedError
    
    
    @property
    def embedding(self) -> chex.Array:
        return jnp.array([self.size, 1])
    
    @property
    def num_actions(self) -> int:
        return self.n_actions
