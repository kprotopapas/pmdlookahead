from environments.gridworlds import GridworldEnv
from policies.classic import REINFORCE, PQA, EQA
import argparse

if __name__ == '__main__':
    n_agents = 20
    size = 16
    horizon = 100
    env_class = GridworldEnv
    alg = EQA
    gamma = 0.9
    epochs = 1000
    lr = 0.05

    parser = argparse.ArgumentParser(prog='PMD TS runner', description='Runner for PMD TS', epilog='')
    parser.add_argument('--n_agents', '-n', type=int, default=100)
    parser.add_argument('--env_name', '-e', type=str, default='Gridworld')
    parser.add_argument('--horizon', '-t', type=int, default=100)
    parser.add_argument('--alg', '-a', type=str, default='reinforce')
    parser.add_argument('--gamma', '-g', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('-l', type=float, default=0.05)


    best_reward = -100000000
    for agent in range(n_agents):
        env = env_class(size=size, horizon=horizon)
        algorithm = alg(env, gamma=0.9)
        rewards = algorithm.learn(epochs=epochs, lr=lr)
        if sum(rewards[-10:]) > best_reward:
            best_reward = sum(rewards[-10:])
            algorithm.pi.save(f'weights/Gridworld-{size**2}-{env_class.__name__}')