import gymnasium as gym
from gymnasium.spaces import Discrete, Dict
import numpy as np


class Catch(gym.Env):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.observation_space = Dict({'paddle': Discrete(width), 'block': Discrete(width), 't': Discrete(height)})
        self.reset()
        
    
    @property
    def state(self):
        return {'block': self.block, 'paddle': self.paddle, 't': self.t}
    
    @state.setter
    def state(self, s):
        self.block, self.paddle, self.t = s['block'], s['paddle'], s['t']


    def reset(self):
        self.t = 0
        self.paddle = np.random.choice(self.width)
        self.block = np.random.choice(self.width)
        return self.state, {'t': self.t}
    

    def step(self, action):
        mv = 2 * action - 1

        self.paddle = np.clip(mv + self.paddle, 0, self.width - 1)
        self.t += 1

        if self.t == self.height - 1:
            r = 1 * (self.paddle == self.block)
            self.reset()
        else:
            r = 0
        
        return self.state, r, {'t': self.t}


    
    def __str__(self):
        grid = np.zeros((self.height, self.width), dtype=str)
        grid[self.t, self.block] = 'X'
        grid[-1, self.paddle] = '_'
        
        return str(grid)