from environments.gridworlds import GridworldEnv
from policies.classic import REINFORCE
import os

if __name__ == '__main__':
    n_agents = 100
    size = 16
    horizon = 100
    env_class = GridworldEnv
    alg = REINFORCE
    gamma = 0.9
    epochs = 1000
    lr = 0.05

    for agent in range(n_agents):
        env = env_class(size=size, horizon=horizon)
        ppg = alg(env, gamma=0.9)
        rewards = ppg.learn(epochs=epochs, lr=lr)
        ppg.pi.save(f'Gridworld-{size}.npy')