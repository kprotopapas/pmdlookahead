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
    def __init__(self, P: chex.Array, R: chex.Array, mu: chex.Array, gamma: float, name: str):
        super().__init__()
        self.num_states = P.shape[0]
        self._n_actions = P.shape[2]
        self.P = P
        self.R = R
        self.mu = mu
        self.gamma = gamma
        self.display_name = name
        self.num_obs = None


    @property
    def embedding(self) -> chex.Array:
        return jnp.ones((1,))
    
    @partial(jax.jit, static_argnums=(0,))
    def inverse(self, state_ind: int) -> tuple[chex.Array, EnvState]:
        raise NotImplementedError
    
    @partial(jax.jit, static_argnums=(0,))
    def index(self, state: EnvState) -> int:
        raise NotImplementedError
    
    @property
    def num_actions(self) -> int:
        return self._n_actions

    
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    

    @partial(jax.jit, static_argnums=(0,))
    def value(self, probas: chex.Array) -> chex.Array:
        gamma = self.gamma
        P_pi = jnp.sum(probas * self.P, axis=2)
        M = jnp.identity(self.num_states) - gamma * P_pi.T
        r_pi = jnp.sum(probas * self.R, axis=1)

        V = jnp.linalg.solve(M, r_pi)

        next_V = jnp.sum(self.P * V[..., jnp.newaxis, jnp.newaxis], axis=0)
        Q = self.R + gamma * next_V

        return Q
    
    
    @partial(jax.jit, static_argnums=(0,))
    def Q(self, pi: JPolicy) -> chex.Array:
        probas = pi.probas
        return self.value(probas)


    @partial(jax.jit, static_argnums=(0,))
    def V(self, pi: JPolicy) -> chex.Array:
        Q = self.Q(pi)
        V = jnp.sum(pi.probas * Q, axis=1)
        return V
    

    @partial(jax.jit, static_argnums=(0,))
    def optimal(self):
        Q = jnp.zeros_like(self.R)

        probas = 1 * (Q == Q.max(axis=1)[..., jnp.newaxis])
        probas = probas / probas.sum(axis=1)[..., jnp.newaxis]
        Q_new = self.value(probas)
        
        def cond(a: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.bool_:
            Q, Q_new = a
            return ~jnp.allclose(Q, Q_new)

        def step(a: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
            Q, Q_new = a
            Q = Q_new
            probas = 1 * (Q == Q.max(axis=1)[..., jnp.newaxis])
            probas = probas / probas.sum(axis=1)[..., jnp.newaxis]
            Q_new = self.value(probas)
            return Q, Q_new

        Q, Q_new = jax.lax.while_loop(cond, step, (Q, Q_new))

        return Q_new
    
    # Bellman maximisation operator (VI operator)
    @partial(jax.jit, static_argnums=(0,))
    def bellman(self, V):
        next_V = jnp.sum(self.P * V[..., jnp.newaxis, jnp.newaxis], axis=0)
        Q = self.R + self.gamma * next_V
        TV = jnp.max(Q, axis=1)
        return TV