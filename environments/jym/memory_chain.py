from chex._src.pytypes import Array
import gymnax
from functools import partial
from gymnax.environments.environment import EnvState
from jax import numpy as jnp
import jax
import chex
from gymnax.environments.bsuite.memory_chain import EnvParams, EnvState


from environments.jym import JTabular


# Wrapper for the MemoryChain environment with explicit P, R, µ matrices and without state resets at the end of episode
class MemoryChain(JTabular):
    def __init__(self, gamma: float, num_bits: int = 1, memory_length: int = 5):
        env, params = gymnax.make('MemoryChain-bsuite')
        self.env = env
        self.params: EnvParams = params.replace(memory_length=memory_length)
        num_states = 2**num_bits * num_bits * params.memory_length + 1 # 2^num_bits contexts * num_bits queries * memory_length timesteps + terminal
        
        
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
        actions = jnp.arange(env.num_actions)

        over_states = jax.vmap(step_state, (0, None), 1)
        over_both = jax.vmap(over_states, (None, 0), 2)
        P = over_both(states, actions)

        over_states = jax.vmap(step_reward, (0, None), 0)
        over_both = jax.vmap(over_states, (None, 0), 1)
        R = over_both(states, actions)


        mu = jnp.zeros((num_states, )) # TODO

        name = f'Catch-Wrapper'
        super().__init__(P, R, mu, gamma, name)
        self.config['num_bits'] = num_bits
        self.config['memory_length'] = memory_length
        self.print_name = f'{name}-{memory_length}-{num_bits}-{gamma}'


    @partial(jax.jit, static_argnums=(0, ))
    def step(self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams):
        return self.env.step_env(key, state, action, params)
    

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, key: chex.PRNGKey, params: EnvParams):
        return self.env.reset_env(key, params)


    def get_obs(self, state: EnvState) -> Array:
        return self.env.get_obs(state)

    @property
    def embedding(self) -> chex.Array:
        return 1
    
    @partial(jax.jit, static_argnums=(0,))
    def inverse(self, state_ind: int) -> tuple[chex.Array, EnvState]:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))    
    def index(self, state: EnvState) -> int:
        context = state.context
        query = state.query
        time = state.time
        terminal = time > self.params.memory_length
        
    

    @property
    def default_params(self):
        return self.params