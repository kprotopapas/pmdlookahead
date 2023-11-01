from environments.discrete import GridworldEnv
import numpy as np
from helper import *

size = 6
gw = GridworldEnv(size=size)
print(gw)

while True:
    action = int(input())
    print()
    state, r, _ = gw.step(action)
    print(gw)
    print(f'Reward: {r}, State: {state}')
    print()