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
        env, params = gymnax.make('DeepSea-bsuite', size=size)
        params = params.replace(deterministic=deterministic)
        self.size = size
        self.env = env
        self.params = params

        prb_swp = 1 / self.size

        def step(s, a):
            # nondeterministic transition: left -> left, right -> 1 / size * left + (1 - 1 / size) * right
            dx = 2 * a - 1
            row, col = s // 64, s % 64
            sp_det = (row + 1) * 64 + jnp.clip(col + dx, 0, self.size - 1)
            sp_swp = (row + 1) * 64 + jnp.clip(col - dx, 0, self.size - 1) 
            
            base = jnp.zeros((size**2 + 1, ), dtype=jnp.float32)
            sp_oh_det = base.at[sp_det].set(1)
            sp_oh_swp = base.at[sp_det].add(1 - prb_swp).at[sp_swp].add(prb_swp)

            return jax.lax.select(deterministic, sp_oh_det, sp_oh_swp)
        

        states = jnp.arange(size**2 + 1)
        actions = jnp.arange(2)

        over_states = jax.vmap(step, (0, None), 1)
        over_both = jax.vmap(over_states, (None, 0), 2)
        P = over_both(states, actions)

        penalty = -0.01
        R = jnp.zeros((size**2 + 1, 2)).at[:, 1].set(penalty / size)
        R = R.at[size**2 - 1, 1].set(1)
        R = R.at[size**2, :].set(0)
        mu = jnp.zeros((size**2 + 1, )).at[0].set(1)

        
        super().__init__(P, R, mu, gamma, 'DeepSea-Wrapper')


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
        row, col = jnp.argwhere(obs == 1, size=1, fill_value=self.size)[0]
        not_terminal = row < self.size
        return jax.lax.select(not_terminal, row * self.size + col, self.size**2)
    

    @property
    def default_params(self):
        return self.params