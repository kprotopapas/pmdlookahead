from __future__ import annotations


import chex
from jax import numpy as jnp
import numpy as np

State = chex.Array
Reward = chex.Array
Done = chex.Array
Action = chex.Array

@chex.dataclass(kw_only=True)
class JEnv(object):
    state: State
    n_actions: int
    gamma: float
    done: Done = jnp.bool_(False)
    reward: Reward = jnp.float32(0)
    t: int = 0

    def step(self, action: Action) -> tuple[JEnv, Reward, Done]:
        pass
    
    def reset(self, _ = None) -> JEnv:
        pass


@chex.dataclass(kw_only=True)
class JDeepsea(JEnv):
    size: int
    state: State = jnp.zeros((2, ), dtype=jnp.int32)
    n_actions: int = 2


    def step(self, action: Action) -> tuple[JEnv, Reward, Done]:
        env = self
        position = env.state
        action = jnp.clip(action, 0, 1)
        dx = jnp.where(env.done, 0, action * 2 - 1)

        temp = jnp.array([position[0] + 1, position[1] + dx])

        newpos = jnp.clip(temp, 0, env.size - 1)
        done = newpos[0] == env.size - 1
        right_penalty = -0.1 * env.gamma**env.size / env.size

        reward = action * jnp.where(newpos[1] == env.size - 1, 1, right_penalty)
        reward = jnp.where(env.done, 0, reward)
        cumulative = env.reward + reward * self.gamma**self.t
        self.t += 1

        return JDeepsea(state=newpos,
                        reward=cumulative,
                        done=done,
                        size=env.size,
                        gamma=self.gamma), reward, done
    
    
    
    def reset(self, _ = None):
        return JDeepsea(size = self.size, gamma = self.gamma)
    

    def __str__(self):
        env = self
        grid = np.arange(env.size**2).reshape((env.size, env.size)).astype(str)
        grid[env.state[0], env.state[1]] = '#'
        grid[-1, -1] = 'X'
        return str(grid)


@chex.dataclass(kw_only=True)
class JTrap(JEnv):
    size: int
    state: State = 0
    n_actions: int = 2

    def reward_matrix(self):
        end = jnp.array([1], dtype=jnp.float32)
        multiplier = self.gamma**self.size / self.size
        ones = jnp.ones((self.size - 1), dtype=jnp.float32)
        penalty = ones * multiplier
        middle = jnp.array([0], dtype=jnp.float32)

        reward_matrix: chex.Array = jnp.hstack([
            end,
            -penalty,
            middle,
            penalty,
            end
        ])

        return reward_matrix


    def step(self, action: Action) -> tuple[JEnv, Reward, Done]:
        position = self.state
        action = jnp.clip(action, 0, 1)
        dx = jnp.where(self.done, 0, action * 2 - 1)

        newpos = jnp.clip(self.state + dx, -self.size, self.size)
        length = 2 * self.size + 1
        done = (position == 0) | (position == length - 1) | self.done
        reward = jnp.where(done, 0, self.reward_matrix()[newpos + self.size])
        cumulative = self.reward + self.gamma**self.t * reward
        self.t += 1


        return JTrap(state=newpos,
                        reward=cumulative,
                        done=done,
                        size=self.size,
                        gamma=self.gamma), reward, done
    
    
    def reset(self, _ = None):
        return JTrap(size = self.size, gamma = self.gamma)
    

    def __str__(self):
        env = self
        board = np.array(self.reward_matrix).astype(str)
        board[env.state] = '#'
        return str(board)


class JPolicy(object):
    # Returns logits associated with current 
    def act(self, env: JEnv) -> chex.Array:
        pass


class JSoftmaxPolicy(JPolicy): # softmax policy
    def __init__(self, n_states: int, n_act: int, embedding = lambda x: x):
        self.embedding = embedding
        self.logits = jnp.zeros((n_states, n_act), dtype=jnp.float32)

    def act(self, env: JEnv) -> chex.Array:
        state = self.embedding(env.state)
        return self.logits[state]