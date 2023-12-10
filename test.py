from environments.solvable import *

env = Deepsea(5, 0.9)
print(env)

while True:
    action = int(input())
    print(env.step(action))
    print(env)