import gymnasium as gym
from environments.discrete import DiscreteStochastic
from helper import np
from baselines.policies import DirectPolicy, Policy, SoftmaxPolicy
from scipy.stats import entropy

import numpy as np
import wandb

from typing import Callable


class Algorithm(object):
    def __init__(self, env: DiscreteStochastic, gamma: float = 0.9, horizon: int = 100, name: str = None):
        self.env = env
        self.gamma = gamma
        self.horizon = horizon

        self.name = name
        self.k = 0
        self.samples = 0
        self.config = {
                'algorithm': self.name,
                'environment': self.env.name,
                'horizon': self.horizon
            }
        
        self.pi: Policy = None


    def learn(self, epochs: int = 1000, lr: Callable | float = 0.1, exact=False, use_wandb=True):
        info = []
        if use_wandb:
            config = {
                    'learning_rate': 'custom' if callable(lr) else lr,
                    'epochs': epochs,
                    'exact': exact
                }
            for key, value in self.config.items():
                config[key] = value

        
            wandb.init(
                # set the wandb project where this run will be logged
                project="pmd-ts",
                # track hyperparameters and run metadata
                config=self.config
            )

        V_opt = self.env.optimal(self.gamma).max(axis=1)

        for k in range(epochs):
            Q = self.env.value(self.pi.probas, self.gamma)
            grad_exact = self.grad_exact(Q)
            grad_est = exact or self.grad()

            gradient = grad_exact if exact else grad_est
            lrk = lr(k, gradient) if callable(lr) else self.lrk(lr, k, gradient)
            if lrk == np.inf:
                pass
            if lrk > 100:
                pass
            self.pi.update(gradient, lrk)

            V = np.sum(self.pi.probas * Q, axis=1)
            sub = np.max(np.abs(V_opt - V))

            # A = Q - Q.mean(axis=1)[..., np.newaxis]
            A_est = gradient - gradient.mean(axis=1)[..., np.newaxis]
            err = gradient - grad_exact
            err = err - err.mean(axis=1)[..., np.newaxis] # for simplex or exp update mean doesn't matter

            out_log = {'epoch': k,
                           'suboptimality': sub,
                           'gradient_effect': np.max(np.abs(A_est)),
                           'learning_rate': lrk,
                           'pi_entropy': np.mean(entropy(self.pi.probas, axis=1))
                           }
            
            if not exact:
                out_log['normalized_grad_error'] = np.max(np.abs(err))
                out_log['samples'] = self.samples

            info.append(out_log)

            if use_wandb:
                wandb.log(out_log)
                

        if use_wandb:
            wandb.finish()

        return info


    def reward_to_go(self, rewards: [float]):
        discounts = np.power(self.gamma, np.arange(len(rewards)))
        discounted = rewards * discounts
        rtg = np.cumsum(discounted[::-1])[::-1] / discounts
        return rtg


    def sample_trajectory(self, state = None, action = None):
        reward_list = []
        state_list = []
        action_list = []

        old_state, info = self.env.reset()
        if state == None:
            self.env.state = state
        else:
            state = old_state

        state_list.append(state)

        terminal = 'terminal' in info and info['terminal']
        
        start = 0
        if action is not None:
            state, reward, info = self.env.step(action)
            reward_list.append(reward)
            action_list.append(action)
            state_list.append(state)
            start = 1
            
        for t in range(start, self.horizon):
            terminal = 'terminal' in info and info['terminal']
            if terminal:
                break
            action = self.pi.sample(state)
            state, reward, info = self.env.step(action)
            reward_list.append(reward)
            action_list.append(action)
            state_list.append(state)


        return np.array(state_list), np.array(action_list), np.array(reward_list)
    

    def grad(self):
        raise NotImplementedError

    def grad_exact(self, Q):
        raise NotImplementedError

    def lrk(self, lr, k, grad):
        return lr


class REINFORCE(Algorithm):
    # Projected Policy Gradient Ascent using REINFORCE
    def __init__(self, env: DiscreteStochastic, gamma: float = 0.9, horizon: int = 100):
        super().__init__(env, gamma, horizon, 'REINFORCE-PPG')
        self.pi = DirectPolicy(self.env.observation_space, self.env.action_space)


    def grad(self):
        states, actions, rewards = self.sample_trajectory()
        rtg = self.reward_to_go(rewards)
        gradient = np.zeros_like(self.pi.probas)

        for t, z in enumerate(zip(states[:-1], actions, rtg)):
            s, a, r = z
            gradient[s, a] += self.gamma**t * r / self.pi.probas[s, a]

        return gradient


    def grad_exact(self, Q): # need to calculate d^\pi somehow...
        pass # TODO


class PQA(Algorithm):
    # Projected Q-Ascent using MC rollouts to estimate Q
    def __init__(self, env: DiscreteStochastic, gamma: float = 0.9, horizon: int = 100):
        super().__init__(env, gamma, horizon, 'PQA')
        self.pi = DirectPolicy(self.env.observation_space, self.env.action_space)


    def grad(self):
        states, actions, rewards = self.sample_trajectory()
        rtg = self.reward_to_go(rewards)

        gradient = np.zeros_like(self.pi.probas)

        for t, z in enumerate(zip(states[:-1], actions, rtg)):
            s, a, r = z
            gradient[s, a] += self.gamma**t * r

        return gradient


    def grad_exact(self, Q):
        return Q
    
    def lrk(self, lr, k, gradient):
        return lr / self.gamma**(2*k + 1)
    

class EQA(Algorithm):
    # Exponentiated Q-Ascent (NPG) using MC rollouts to estimate Q
    def __init__(self, env: DiscreteStochastic, gamma: float = 0.9, horizon: int = None, n_trajectories: int = None):
        if horizon is None:
            horizon = min(int(np.ceil(1 / (1 - gamma))), 100)

        if n_trajectories is None:
            n_trajectories = min(int(np.ceil(1 / gamma**(2 * horizon))), 10)
        self.n_trajectories = n_trajectories

        super().__init__(env, gamma, horizon, 'EQA')

        self.config['n_trajectories'] = n_trajectories
        self.pi = SoftmaxPolicy(self.env.observation_space, self.env.action_space)



    def grad(self):
        Q = np.zeros_like(self.env.R)

        for s in range(self.env.n_obs):
            for a in range(self.env.n_act):
                for _ in range(self.n_trajectories):
                    states, actions, rewards = self.sample_trajectory(s, a)
                    rtg = self.reward_to_go(rewards)
                    Q[s, a] += rtg[0]

        Q = Q / self.n_trajectories
        self.samples += self.horizon * self.n_trajectories
        return Q


    def grad_exact(self, Q):
        return Q


    def lrk(self, lr, k, gradient):
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = np.max(rel_entr(greedy, pk).sum(axis=1))

        lrk = lr * dk / self.gamma**(2*(k + 1))
        if lrk == np.inf:
            pass
        return lrk


class DEQA(Algorithm):
    # Exponentiated Q-Ascent (NPG) using d-step lookahead to estimate Q
    def __init__(self, env: DiscreteStochastic, gamma: float = 0.9, horizon: int = 100, d: int = 3):
        self.d = d
        name = 'EQA' if d == 1 else f'{d}-EQA'
        super().__init__(env, gamma, horizon, name)
        self.pi = SoftmaxPolicy(self.env.observation_space, self.env.action_space)


    def grad(self):
        raise NotImplementedError


    def grad_exact(self, Q):
        if self.d == 1:
            return Q
        
        V = np.sum(self.pi.probas * Q, axis=1)
        tdm = V

        for _ in range(self.d - 1):
            tdm = self.env.bellman(tdm, self.gamma)

        next_Vd = self.env.P.transpose((2, 1, 0)) @ tdm
        Qd = self.env.R + self.gamma * next_Vd.transpose((1, 0))
        return Qd


    def lrk(self, lr, k, gradient):
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = np.max(entropy(greedy, pk, axis=1))

        lrk = lr * dk / self.gamma**(2*(k + 1) * self.d)
        return lrk


class DPQA(Algorithm):
    # Projected Q-Ascent using MC rollouts to estimate Q, with d-step lookahead for Q
    def __init__(self, env: DiscreteStochastic, gamma: float = 0.9, horizon: int = 100, d: int = 3):
        self.d = d
        name = 'PQA' if d == 1 else f'{d}-PQA'
        super().__init__(env, gamma, horizon, name)
        self.pi = DirectPolicy(self.env.observation_space, self.env.action_space)


    def grad(self):
        raise NotImplementedError
    

    def grad_exact(self, Q):
        if self.d == 1:
            return Q
        
        V = np.sum(self.pi.probas * Q, axis=1)
        tdm = V

        for _ in range(self.d - 1):
            tdm = self.env.bellman(tdm, self.gamma)

        next_Vd = self.env.P.transpose((2, 1, 0)) @ tdm
        Qd = self.env.R + self.gamma * next_Vd.transpose((2, 1))
        return Qd
    
    
    def lrk(self, lr, k, gradient):
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = np.linalg.norm(pk - greedy, axis=1).max()

        lrk = lr * dk / self.gamma**(2*(k + 1) * self.d)
        return lrk