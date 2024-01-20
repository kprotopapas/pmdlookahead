from flax import struct

import chex
from jax import numpy as jnp
import jax
from gymnax.environments.environment import Environment
from functools import partial

from algorithms.policies import JPolicy


@struct.dataclass
class EnvState:
    pass


@struct.dataclass
class EnvParams:
    pass


class JTabular(Environment):
    def __init__(self, num_states: int, n_actions: int, gamma: float, name: str):
        super().__init__()
        self.num_states = num_states
        self._n_actions = n_actions
        self.gamma = gamma
        self.display_name = name


    @property
    def embedding(self) -> chex.Array:
        return jnp.ones((1,))
    
    @partial(jax.jit, static_argnums=(0,))
    def inverse(self, state_ind: int) -> tuple[chex.Array, EnvState]:
        raise NotImplementedError
    
    @partial(jax.jit, static_argnums=(0,))
    def index(self, obs: chex.Array) -> int:
        raise NotImplementedError
    
    @property
    def num_actions(self) -> int:
        return self._n_actions