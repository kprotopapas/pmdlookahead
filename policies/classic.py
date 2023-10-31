import numpy as np
import wandb
import gymnasium as gym

from helper import *
from environments.gridworlds import GridworldEnv


class Policy:
    def sample(self, state):
        raise NotImplementedError
    
    def logprob(self, state, actions):
        raise NotImplementedError
    
    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class DirectPolicy(Policy):
    def __init__(self, states, actions):
        n_states = states.n
        n_actions = actions.n

        self.n_states = n_states
        self.n_actions = n_actions
        self.probabilities = np.ones((n_states, n_actions)) / n_actions
        self.actions = actions
    

    def sample(self, state):
        action = np.random.choice(np.arange(self.n_actions), p=self.probabilities[state])
        return action


    def update(self, grad, lr: float):
        new = self.probabilities + lr * grad
        for i in range(self.probabilities.shape[0]):
            projected = simplex_project_sort(new[i])
            self.probabilities[i] = projected

    
    def save(self, path):
        with open(path, 'wb') as f:
            np.save(f, self.probabilities, allow_pickle=True)

    
    def load(self, path):
        with open(path, 'rb') as f:
            self.probabilities = np.load(f, allow_pickle=True)

    
def reward_to_go(rewards: [float], discount: float):
    discounts = np.power(discount, np.arange(len(rewards)))
    discounted = rewards * discounts
    rtg = np.cumsum(discounted[::-1])[::-1] / discounts
    return rtg


def sample_trajectory(env: gym.Env, policy: Policy):
    rewards = []
    states = []
    actions = []

    state, info = env.reset()
    states.append(state)

    terminal = info['terminal']
    
    while not terminal:
        action = policy.sample(state)
        state, reward, info = env.step(action)
        terminal = info['terminal']

        rewards.append(reward)
        actions.append(action)
        states.append(state)

    return np.array(states), np.array(actions), np.array(rewards)


class Algorithm:
    def __init__(self, env: GridworldEnv, gamma: float = 0.9):
        self.env = env
        self.gamma = gamma
        self.pi = None


    def learn(self, epochs: int = 1000, lr: float = 0.05):
        totals = []
        wandb.init(
            # set the wandb project where this run will be logged
            project="pmd-ts",
            
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "algorithm": "REINFORCE-PPG",
                "environment": f"Gridworld-{self.env.size}",
                "epochs": epochs,
            }
        )

        for k in range(epochs):
            states, actions, rewards = sample_trajectory(self.env, self.pi)
            reward = np.sum(rewards)
            totals.append(reward)

            rtg = reward_to_go(rewards, self.gamma)
            gradient = self.grad(states, actions, rtg)
            self.pi.update(gradient, 0.05)

            wandb.log({'reward': reward, 'k': k})

        wandb.finish()
        return totals


    def grad(self, states, actions, rtg):
        raise NotImplementedError


class REINFORCE(Algorithm):
    #Projected Policy Gradient Ascent using REINFORCE
    def __init__(self, env: GridworldEnv, gamma: float = 0.9):
        super().__init__(env, gamma)
        self.gamma = gamma
        self.env = env
        self.pi = DirectPolicy(self.env.observation_space, self.env.action_space)
    

    def grad(self, states, actions, rtg):
        gradient = np.zeros_like(self.pi.probabilities)

        for t, z in enumerate(zip(states[:-1], actions, rtg)):
            s, a, r = z
            gradient[s, a] += self.gamma**t * r / self.pi.probabilities[s, a]

        return gradient