from chex._src.pytypes import Array
from flax import struct
from functools import partial
from gymnax.environments.environment import EnvState
from jax import numpy as jnp
import jax
import chex
from gymnax.environments.bsuite.catch import EnvParams, EnvState


from environments.jym import JTabular


@struct.dataclass
class EnvState:
    pos_x: jnp.int32
    pos_y: jnp.int32



@struct.dataclass
class EnvParams:
    target_x: jnp.int32
    target_y: jnp.int32
    deterministic: jnp.bool_


# Custom Gridworld environment with P, R, µ matrices and without state resets at the end of episode
class Gridworld(JTabular):
    def __init__(self, gamma: float, rows: int = 10, columns: int = 5,
                 deterministic: bool = True, seed: int = 42):

        num_states = rows * columns
        self.rows = rows
        self.columns = columns

        key = jax.random.PRNGKey(seed)
        target_x = jnp.int32(jax.random.uniform(key, shape=(), minval=0, maxval=columns))
        target_y = jnp.int32(jax.random.uniform(key, shape=(), minval=0, maxval=rows))

        params = EnvParams(target_x=target_x, target_y=target_y,
                           deterministic=jnp.bool_(deterministic))
        self.params = params

        
        def step_state(s, a):
            _, state = self.inverse(s)
            key = jax.random.PRNGKey(0)
            _, nstate, _, _, _ = self.step(key, state, a, params.replace(deterministic=True))
            opposite_action = (a + 2) % 45
            _, nswap, _, _, _ = self.step(key, state, opposite_action, params.replace(deterministic=True))
            ns = self.index(nstate)
            nswap = self.index(nswap)
            
            mag = jax.lax.select(a % 2 == 0, self.rows, self.columns)
            non_det = jnp.zeros((num_states, )).at[ns].set(1 - 1 / mag).at[nswap].set(1 / mag)
            det = jnp.zeros((num_states, )).at[ns].set(1)

            return jax.lax.select(deterministic, det, non_det)


        def step_reward(s, a):
            _, state = self.inverse(s)
            key = jax.random.PRNGKey(0)
            _, _, r, _, _ = self.step(key, state, a, params)
            return r
        

        states = jnp.arange(num_states)
        actions = jnp.arange(4)

        over_states = jax.vmap(step_state, (0, None), 1)
        over_both = jax.vmap(over_states, (None, 0), 2)
        P = over_both(states, actions)

        over_states = jax.vmap(step_reward, (0, None), 0)
        over_both = jax.vmap(over_states, (None, 0), 1)
        R = over_both(states, actions)


        target_ind = self.index(EnvState(pos_x=target_x, pos_y=target_y))
        mu = jnp.ones((num_states, )).at[target_ind].set(0) / (num_states - 1)

        name = f'Gridworld'
        super().__init__(P, R, mu, gamma, name)
        self.config['rows'] = rows
        self.config['columns'] = columns
        det = f'rand' if not deterministic else ''
        self.print_name = f'{name}-{rows}x{columns}-{gamma}{det}'


    # action: [0, 1, 2, 3] -> [DOWN, LEFT, UP, RIGHT]
    @partial(jax.jit, static_argnums=(0, ))
    def step(self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams):
        action = jnp.clip(action, 0, 3)
        terminal_x = state.pos_x == self.params.target_x
        terminal_y = state.pos_y == self.params.target_y
        terminal = jnp.logical_and(terminal_x, terminal_y)
        not_terminal = jnp.logical_not(terminal)

        down = jax.lax.select(jnp.logical_and(action == 0, not_terminal), 1, 0)
        left = jax.lax.select(jnp.logical_and(action == 1, not_terminal), 1, 0)
        up = jax.lax.select(jnp.logical_and(action == 2, not_terminal), 1, 0)
        right = jax.lax.select(jnp.logical_and(action == 3, not_terminal), 1, 0)

        dy = down - up
        dx = right - left

        h_key, v_key = jax.random.split(key)
        swap_v = jax.random.bernoulli(v_key, p=1 / self.rows, shape=())
        swap_h = jax.random.bernoulli(h_key, p=1 / self.columns, shape=())

        not_det = jnp.logical_not(params.deterministic)
        fy = jax.lax.select(jnp.logical_and(swap_v, not_det), -dy, dy)
        fx = jax.lax.select(jnp.logical_and(swap_h, not_det), -dx, dx)

        new_x = jnp.clip(state.pos_x + fx, 0, self.columns - 1)
        new_y = jnp.clip(state.pos_y + fy, 0, self.rows - 1)

        new_state = EnvState(pos_x=new_x, pos_y=new_y)
        obs = self.get_obs(new_state)

        reward = jax.lax.select(terminal, 1, 0)

        return obs, new_state, reward, terminal, {}
    
    

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, key: chex.PRNGKey, params: EnvParams):
        index = jax.random.categorical(key, jnp.log(self.mu), shape=())
        obs, state = self.inverse(index)
        return obs, state
    

    def get_obs(self, state: EnvState) -> Array:
        grid = jnp.zeros((self.rows, self.columns))
        grid = grid.at[state.pos_y, state.pos_x].set(1)
        target_x, target_y = self.params.target_x, self.params.target_y
        grid = grid.at[target_y, target_x].set(-1)
        return grid


    @property
    def embedding(self) -> chex.Array:
        return 1
    

    @partial(jax.jit, static_argnums=(0,))
    def inverse(self, state_ind: int) -> tuple[chex.Array, EnvState]:
        pos_y = state_ind // self.columns
        pos_x = state_ind % self.columns

        state = EnvState(pos_x=pos_x, pos_y=pos_y)
        obs = self.get_obs(state)
        return obs, state


    @partial(jax.jit, static_argnums=(0,))    
    def index(self, state: EnvState) -> int:
        return state.pos_x + state.pos_y * self.columns # number in base (columns)
    

    @property
    def default_params(self):
        return self.params