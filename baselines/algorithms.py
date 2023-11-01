import gymnasium as gym
from environments.gridworlds import GridworldEnv
from helper import np
from baselines.policies import DirectPolicy, Policy, SoftmaxPolicy

import numpy as np
import wandb

from typing import Callable


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
        self.pi: Policy = None
        self.name = None


    def learn(self, epochs: int = 1000, lr: Callable | float = 0.1, exact=False):
        totals = []
        wandb.init(
            # set the wandb project where this run will be logged
            project="pmd-ts",
            # track hyperparameters and run metadata
            config={
                'learning_rate': lr,
                'algorithm': self.name,
                'environment': self.env.name,
                'epochs': epochs,
                'exact': exact
            }
        )

        for k in range(epochs):
            if k == epochs - 1:
                pass
            if exact:
                Q = self.env.value(self.pi.probas, self.gamma)
                gradient = self.grad_exact(Q)
                reward = np.min(np.sum(self.pi.probas * Q, axis=1))

            else:
                states, actions, rewards = sample_trajectory(self.env, self.pi)
                reward = np.sum(rewards)
                totals.append(reward)

                rtg = reward_to_go(rewards, self.gamma)
                gradient = self.grad(states, actions, rtg)


            lrk = lr(k) if callable(lr) else self.lrk(lr, k)
            self.pi.update(gradient, lrk)

            worst_steps = self.env.size * 2 - 2
            best_worst = np.sum([self.gamma**i for i in range(worst_steps - 1)])

            wandb.log({'reward': reward, 'k': k, 'suboptimality': best_worst + reward})

        wandb.finish()
        return totals


    def grad(self, states, actions, rtg, exact=False):
        raise NotImplementedError

    def grad_exact(self, Q):
        raise NotImplementedError

    def lrk(self, lr, k):
        return lr


class REINFORCE(Algorithm):
    # Projected Policy Gradient Ascent using REINFORCE
    def __init__(self, env: GridworldEnv, gamma: float = 0.9):
        super().__init__(env, gamma)
        self.gamma = gamma
        self.env = env
        self.pi = DirectPolicy(self.env.observation_space, self.env.action_space)
        self.name = 'REINFORCE-PPG'


    def grad(self, states, actions, rtg):
        gradient = np.zeros_like(self.pi.probabilities)

        for t, z in enumerate(zip(states[:-1], actions, rtg)):
            s, a, r = z
            gradient[s, a] += self.gamma**t * r / self.pi.probabilities[s, a]

        return gradient


    def grad_exact(self, Q):
        pass # TODO


class PQA(Algorithm):
    # Projected Q-Ascent using MC rollouts to estimate Q
    def __init__(self, env: GridworldEnv, gamma: float = 0.9):
        super().__init__(env, gamma)
        self.gamma = gamma
        self.env = env
        self.pi = DirectPolicy(self.env.observation_space, self.env.action_space)
        self.name = 'PQA'


    def grad(self, states, actions, rtg): # TODO implement exact
        gradient = np.zeros_like(self.pi.probabilities)

        for t, z in enumerate(zip(states[:-1], actions, rtg)):
            s, a, r = z
            gradient[s, a] += self.gamma**t * r

        return gradient


    def grad_exact(self, Q):
        return Q


class EQA(Algorithm):
    # Exponentiated Q-Ascent (NPG) using MC rollouts to estimate Q
    def __init__(self, env: GridworldEnv, gamma: float = 0.9):
        super().__init__(env, gamma)
        self.gamma = gamma
        self.env = env
        self.pi = SoftmaxPolicy(self.env.observation_space, self.env.action_space)
        self.name = 'EQA'


    def grad(self, states, actions, rtg):
        gradient = np.zeros_like(self.pi.params)

        for t, z in enumerate(zip(states[:-1], actions, rtg)):
            s, a, r = z
            gradient[s, a] += self.gamma**t * r

        return gradient


    def grad_exact(self, Q):
        return Q


    def lrk(self, lr, k):
        return lr / self.gamma**k


class DEQA(Algorithm):
    # Exponentiated Q-Ascent (NPG) using d-step lookahead to estimate Q
    def __init__(self, env: GridworldEnv, gamma: float = 0.9, d: int = 3):
        super().__init__(env, gamma)
        self.gamma = gamma
        self.env = env
        self.d = d
        self.pi = SoftmaxPolicy(self.env.observation_space, self.env.action_space)
        self.name = f'{d}-EQA'


    def grad(self, states, actions, rtg):
        gradient = np.zeros_like(self.pi.params)

        for t, z in enumerate(zip(states[:-1], actions, rtg)):
            s, a, r = z
            gradient[s, a] += self.gamma**t * r

        return gradient


    def grad_exact(self, Q):
        V = np.sum(self.pi.probas * Q, axis=1)
        tdm = V

        for _ in range(self.d - 1):
            tdm = self.env.bellman(tdm, self.gamma)

        next_Vd = np.sum(self.env.P * tdm.reshape((self.env.n_obs, 1, 1)), axis=0)
        Qd = self.env.R + self.gamma * next_Vd
        return Qd


    def lrk(self, lr, k):
        return lr / self.gamma**(k * self.d)