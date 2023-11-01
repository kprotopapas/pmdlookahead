import gymnasium as gym
from environments.discrete import DiscreteStochastic
from helper import np
from baselines.policies import DirectPolicy, Policy, SoftmaxPolicy

import numpy as np
import wandb

from typing import Callable


class Algorithm:
    def __init__(self, env: DiscreteStochastic, gamma: float = 0.9, horizon: int = 100):
        self.env = env
        self.gamma = gamma
        self.horizon = horizon

        self.pi: Policy = None
        self.name = None


    def learn(self, epochs: int = 1000, lr: Callable | float = 0.1, exact=False):
        totals = []
        wandb.init(
            # set the wandb project where this run will be logged
            project="pmd-ts",
            # track hyperparameters and run metadata
            config={
                'learning_rate': 'adaptive' if callable(lr) else lr,
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
                states, actions, rewards = self.sample_trajectory()
                rtg = self.reward_to_go(rewards)
                gradient = self.grad(states, actions, rtg)

                reward = rtg[0]
                totals.append(reward)


            lrk = lr(k) if callable(lr) else self.lrk(lr, k)
            self.pi.update(gradient, lrk)

            optimal = self.env.opt
            best_worst = np.sum([self.gamma**i for i in range(optimal - 1)])

            wandb.log({'reward': reward, 'k': k, 'suboptimality': best_worst + reward})

        wandb.finish()
        return totals


    def reward_to_go(self, rewards: [float]):
        discounts = np.power(self.gamma, np.arange(len(rewards)))
        discounted = rewards * discounts
        rtg = np.cumsum(discounted[::-1])[::-1] / discounts
        return rtg


    def sample_trajectory(self, state = None):
        rewards = []
        states = []
        actions = []

        old_state, info = self.env.reset()
        if state == None:
            self.env.state = state
        else:
            state = old_state

        states.append(state)

        terminal = 'terminal' in info and info['terminal']
        for t in range(self.horizon):
            action = self.pi.sample(state)
            state, reward, info = self.env.step(action)
            terminal = 'terminal' in info and info['terminal']
            if terminal:
                break

            rewards.append(reward)
            actions.append(action)
            states.append(state)

        return np.array(states), np.array(actions), np.array(rewards)
    

    def grad(self, states, actions, rtg, exact=False):
        raise NotImplementedError

    def grad_exact(self, Q):
        raise NotImplementedError

    def lrk(self, lr, k):
        return lr


class REINFORCE(Algorithm):
    # Projected Policy Gradient Ascent using REINFORCE
    def __init__(self, env: DiscreteStochastic, gamma: float = 0.9, horizon: int = 100):
        super().__init__(env, gamma, horizon)
        self.pi = DirectPolicy(self.env.observation_space, self.env.action_space)
        self.name = 'REINFORCE-PPG'


    def grad(self, states, actions, rtg):
        gradient = np.zeros_like(self.pi.probas)

        for t, z in enumerate(zip(states[:-1], actions, rtg)):
            s, a, r = z
            gradient[s, a] += self.gamma**t * r / self.pi.probas[s, a]

        return gradient


    def grad_exact(self, Q):
        pass # TODO


class PQA(Algorithm):
    # Projected Q-Ascent using MC rollouts to estimate Q
    def __init__(self, env: DiscreteStochastic, gamma: float = 0.9, horizon: int = 100):
        super().__init__(env, gamma, horizon)
        self.pi = DirectPolicy(self.env.observation_space, self.env.action_space)
        self.name = 'PQA'


    def grad(self, states, actions, rtg):
        gradient = np.zeros_like(self.pi.probas)

        for t, z in enumerate(zip(states[:-1], actions, rtg)):
            s, a, r = z
            gradient[s, a] += self.gamma**t * r

        return gradient


    def grad_exact(self, Q):
        return Q


class EQA(Algorithm):
    # Exponentiated Q-Ascent (NPG) using MC rollouts to estimate Q
    def __init__(self, env: DiscreteStochastic, gamma: float = 0.9, horizon: int = 100):
        super().__init__(env, gamma, horizon)
        self.pi = SoftmaxPolicy(self.env.observation_space, self.env.action_space)
        self.name = 'EQA'


    def grad(self, states, actions, rtg):
        gradient = np.zeros_like(self.pi.probas)

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
    def __init__(self, env: DiscreteStochastic, gamma: float = 0.9, horizon: int = 100, d: int = 3):
        super().__init__(env, gamma, horizon)
        self.d = d
        self.pi = SoftmaxPolicy(self.env.observation_space, self.env.action_space)
        self.name = f'{d}-EQA'


    def grad(self, states, actions, rtg):
        gradient = np.zeros_like(self.pi.probas)

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