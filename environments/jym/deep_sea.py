import gymnax
from functools import partial
from jax import numpy as jnp
import jax
import chex
from gymnax.environments.bsuite.deep_sea import EnvParams, EnvState


from environments.jym import JTabular


# Wrapper for the DeepSea environment with explicit P, R, µ matrices and without state resets at the end of episode
class DeepSea(JTabular):
    def __init__(self, size: int, gamma: float, deterministic: bool = True):
        env, params = gymnax.make('DeepSea-bsuite', size=size)
        params = params.replace(deterministic=deterministic)
        self.size = size
        self.env = env
        self.params = params
        self.n_obs = self.size**2

        prb_swp = 1 / self.size
        num_states = self.size * (self.size + 1) // 2 + 1

        def step(s, a):
            # nondeterministic transition: left -> left, right -> 1 / size * left + (1 - 1 / size) * right
            row = jnp.floor((jnp.sqrt(8 * s + 1) - 1) / 2).astype(jnp.int32) # returns closest triangle root
            tri = row * (row + 1) // 2
            col = s - tri

            nrow = jnp.minimum(row + 1, self.size).astype(jnp.int32)

            dx = 2 * a - 1
            ncol_det = jax.lax.select(row >= self.size - 1, 0, jnp.clip(col + dx, 0, self.size - 1))
            ncol_swp = jax.lax.select(row >= self.size - 1, 0, jnp.clip(col - dx, 0, self.size - 1))

            ntri = nrow * (nrow + 1) // 2
            sp_det = ntri + ncol_det
            
            sp_swp = ntri + ncol_swp
            
            base = jnp.zeros((num_states, ), dtype=jnp.float32)
            sp_oh_det = base.at[sp_det].set(1)
            sp_oh_swp = base.at[sp_det].add(1 - prb_swp).at[sp_swp].add(prb_swp)

            return jax.lax.select(deterministic, sp_oh_det, sp_oh_swp)
        

        states = jnp.arange(num_states)
        actions = jnp.arange(2)

        over_states = jax.vmap(step, (0, None), 1)
        over_both = jax.vmap(over_states, (None, 0), 2)
        P = over_both(states, actions)

        penalty = -0.01 / size
        R = jnp.zeros((num_states, 2)).at[:, 1].set(penalty)
        
        R = R.at[num_states - 2, 1].set(1 + penalty)
        R = R.at[num_states - 1, :].set(0)
        mu = jnp.zeros((num_states, )).at[0].set(1)

        name = 'DeepSea-Wrapper' + (f'-rand' if not deterministic else '')
        super().__init__(P, R, mu, gamma, name)


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
        row = jnp.floor((jnp.sqrt(8 * state_ind + 1) - 1) / 2).astype(jnp.int32) # returns closest triangle root
        tri = row * (row + 1) // 2
        col = jax.lax.select(row >= self.size, 0, state_ind - tri)

        obs = jnp.zeros((self.size, self.size)).at[row, col].set(1)
        state = EnvState(row, col, row > col, 0, 0, 0.99, jnp.ones_like(obs), row)
        return obs, state

    
    @partial(jax.jit, static_argnums=(0,))
    def index(self, state: EnvState) -> int:
        row, col = state.row, state.column
        tri = row * (row + 1) // 2
        col = jax.lax.select(row >= self.size, 0, col)
        return tri + col
    

    @property
    def default_params(self):
        return self.params