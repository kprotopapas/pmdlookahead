from environments.gridworlds import GridworldEnv
from policies.classic import REINFORCE, PQA, EQA
import argparse

if __name__ == '__main__':
    n_agents = 1100
    size = 32
    horizon = 100
    env_class = GridworldEnv
    alg = EQA
    gamma = 0.99
    epochs = 500
    lr = 0.05
    exact = True

    parser = argparse.ArgumentParser(prog='PMD TS runner', description='Runner for PMD TS', epilog='')
    parser.add_argument('--n_agents', '-n', type=int, default=100)
    parser.add_argument('--env_name', '-e', type=str, default='Gridworld')
    parser.add_argument('--horizon', '-t', type=int, default=100)
    parser.add_argument('--alg', '-a', type=str, default='reinforce')
    parser.add_argument('--gamma', '-g', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('-l', type=float, default=0.05)
    parser.add_argument('--exact', action='store_true')


    best_reward = -100000000
    for agent in range(n_agents):
        env = env_class(size=size, horizon=horizon)
        algorithm = PQA(env, gamma=gamma)
        rewards = algorithm.learn(epochs=epochs, lr=lr, exact=True)
        if sum(rewards[-10:]) > best_reward:
            best_reward = sum(rewards[-10:])
            algorithm.pi.save(f'weights/Gridworld-{size**2}-{alg.__name__}.npy')

        
    best_reward = -100000000
    for agent in range(n_agents):
        env = env_class(size=size, horizon=horizon)
        algorithm = EQA(env, gamma=gamma)
        rewards = algorithm.learn(epochs=epochs, lr=lr, exact=True)
        if sum(rewards[-10:]) > best_reward:
            best_reward = sum(rewards[-10:])
            algorithm.pi.save(f'weights/Gridworld-{size**2}-{alg.__name__}.npy')