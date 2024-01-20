import gymnax
from functools import partial
from jax import numpy as jnp
import jax
import chex
from gymnax.environments.bsuite.deep_sea import EnvParams, EnvState


from environments.jym import JTabular


# Wrapper for the DeepSea environment with ability to set state
class DeepSea(JTabular):
    def __init__(self, size: int, gamma: float, deterministic: bool = True):
        super().__init__(size**2, 2, gamma, 'DeepSea-Wrapper')
        env, params = gymnax.make('DeepSea-bsuite', size=size)
        params = params.replace(deterministic=deterministic)
        self.size = size
        self.env = env
        self.params = params
        self.gamma = gamma

    

    @partial(jax.jit, static_argnums=(0, ))
    def step(self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams):
        return self.env.step_env(key, state, action, params)
    

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, key: chex.PRNGKey, params: EnvParams):
        return self.env.reset_env(key, params)


    @property
    def embedding(self) -> chex.Array:
        return jnp.arange(self.size**2).reshape((self.size, self.size))
    
    @partial(jax.jit, static_argnums=(0,))
    def inverse(self, state_ind: int) -> tuple[chex.Array, EnvState]:
        row = state_ind // self.size
        col = state_ind % self.size

        obs = jnp.zeros((self.size, self.size)).at[row, col].set(1)
        state = EnvState(row, col, row > col, 0, 0, 0.99, jnp.ones_like(obs), row)
        return obs, state

    
    @partial(jax.jit, static_argnums=(0,))
    def index(self, obs: chex.Array) -> int:
        not_terminal = jnp.any(obs > 0)
        row, col = jnp.argwhere(obs == 1)
        return jnp.select(not_terminal, row * self.size + col, self.size**2)