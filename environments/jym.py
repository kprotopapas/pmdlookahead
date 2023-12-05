from __future__ import annotations


import chex
from jax import numpy as jnp
import numpy as np

State = chex.Array
Reward = chex.Array
Done = chex.Array
Action = chex.Array

@chex.dataclass
class Env(object):
    state: State
    n_actions: int
    gamma: float
    done: Done = jnp.bool_(False)
    reward: Reward = jnp.float32(0)

    def step(self, action: Action) -> tuple[Env, Reward, Done]:
        pass
    
    def reset(self, _ = None) -> Env:
        pass


@chex.dataclass(kw_only=True)
class DeepseaX(Env):
    size: int
    state: State = jnp.zeros((2, ), dtype=jnp.int16)
    n_actions: int = 2


    def step(self, action: Action) -> tuple[Env, Reward, Done]:
        env = self
        position = env.state
        action = jnp.clip(action, 0, 1)
        dx = jnp.where(env.done, 0, action * 2 - 1)
        newpos = jnp.clip(jnp.array([position[0] + 1, position[1] + dx]), 0, env.size - 1)
        done = newpos[0] == env.size - 1
        right_reward = -0.1 * env.gamma**env.size / env.size
        reward = action * jnp.where(newpos[1] == env.size - 1, 1, right_reward)
        reward = jnp.where(env.done, 0, reward)

        return DeepseaX(state=newpos,
                        reward=reward + env.reward,
                        done=done,
                        size=env.size,
                        gamma=self.gamma), reward, done
    
    
    def reset(self, _ = None):
        return DeepseaX(size = self.size, gamma = self.gamma)
    

    def __str__(self):
        env = self
        grid = np.arange(env.size**2).reshape((env.size, env.size)).astype(str)
        grid[env.state[0], env.state[1]] = '#'
        return str(grid)


class Policy(object):
    # Returns logits associated with current 
    def act(self, env: Env) -> chex.Array:
        pass



class SoftmaxPolicy(object): # softmax policy
    def __init__(self, embedding = lambda x: x):


    def act(self, env: Env) -> chex.Array:
