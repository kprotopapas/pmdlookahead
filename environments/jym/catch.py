import gymnax
from functools import partial
from jax import numpy as jnp
import jax
import chex
from gymnax.environments.bsuite.catch import EnvParams, EnvState


from environments.jym import JTabular


# Wrapper for the DeepSea environment with explicit P, R, µ matrices and without state resets at the end of episode
class Catch(JTabular):
    def __init__(self, gamma: float, rows: int = 10, columns: int = 5):
        env, params = gymnax.make('Catch-bsuite', rows=rows, columns=columns)
        self.env = env
        self.params = params
        num_states = columns * columns * rows # terminal states are where paddle is on last row, can recycle them
        
        
        def step_state(s, a):
            _, state = self.inverse(s)
            key = jax.random.PRNGKey(0)
            _, nstate, _, _, _ = env.step_env(key, state, a, params)
            ns = self.index(nstate)

            return jnp.zeros((num_states, )).at[ns].set(1)


        def step_reward(s, a):
            _, state = self.inverse(s)
            key = jax.random.PRNGKey(0)
            _, _, r, _, _ = env.step(key, state, a, params)
            return r
        

        states = jnp.arange(num_states)
        actions = jnp.arange(2)

        over_states = jax.vmap(step_state, (0, None), 1)
        over_both = jax.vmap(over_states, (None, 0), 2)
        P = over_both(states, actions)

        over_states = jax.vmap(step_reward, (0, None), 0)
        over_both = jax.vmap(over_states, (None, 0), 1)
        R = over_both(states, actions)


        start = (columns // 2) * columns
        mu = jnp.zeros((num_states, )).at[start: start + columns].set(1 / columns)

        name = f'Catch-Wrapper'
        super().__init__(P, R, mu, gamma, name)
        self.config['rows'] = rows
        self.config['columns'] = columns
        self.print_name = f'{name}-{rows}x{columns}-{gamma}'


    @partial(jax.jit, static_argnums=(0, ))
    def step(self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams):
        return self.env.step_env(key, state, action, params)
    

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, key: chex.PRNGKey, params: EnvParams):
        return self.env.reset_env(key, params)


    @property
    def embedding(self) -> chex.Array:
        return 1
    
    @partial(jax.jit, static_argnums=(0,))
    def inverse(self, state_ind: int) -> tuple[chex.Array, EnvState]:
        timestep = state_ind // self.env.columns**2

        paddle = (state_ind % self.env.columns**2) // self.env.columns
        ball = state_ind % self.env.columns

        state = EnvState(ball_x=ball, ball_y=timestep, 
                         paddle_x=paddle, paddle_y=self.env.rows - 1, prev_done=False, time=timestep)
        obs = self.env.get_obs(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))    
    def index(self, state: EnvState) -> int:
        nball = state.ball_x
        npaddle = state.paddle_x * self.env.columns
        ntime = jnp.minimum(state.ball_y, self.env.rows - 1) * self.env.columns**2
        return nball + npaddle + ntime
    

    @property
    def default_params(self):
        return self.params