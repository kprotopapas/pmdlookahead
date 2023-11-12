import gymnasium as gym
from environments.discrete import Tabular
from helper import np
from baselines.policies import DirectPolicy, Policy, SoftmaxPolicy
from scipy.stats import entropy

import numpy as np
import wandb

from typing import Callable


class Algorithm(object):
    def __init__(self, env: Tabular, horizon: int, n_trajectories: int, pi: Policy, name: str):
        self.env = env
        self.gamma = env.gamma
        self.horizon = horizon
        self.n_trajectories = n_trajectories

        self.name = name
        self.config = {
                'algorithm': self.name,
                'environment': {'name': self.env.name, 'gamma': self.env.gamma},
                'horizon': self.horizon,
                'n_trajectories': n_trajectories
            }
        
        self.pi: Policy = pi


    def learn(self, epochs: int = 1000, lr: Callable | float = 1, exact=False, use_wandb=True, verbose=False):
        self.exact = exact
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

        V_opt = self.env.optimal().max(axis=1)

        self.env.reset(samples=True)

        for k in range(epochs):
            Q = self.env.Q(self.pi)
            grad_exact = self.grad_exact(Q)

            if exact:
                gradient = grad_exact
            else:
                Q_est = self.estimate_value()
                gradient = self.grad(Q_est)

            lrk, dk = self.lrk(lr, k, gradient)
            self.pi.update(gradient, lrk)

            V = np.sum(self.pi.probas * Q, axis=1)
            sub = np.max(np.abs(V_opt - V))

            # A = Q - Q.mean(axis=1)[..., np.newaxis]
            A_est = gradient - gradient.mean(axis=1)[..., np.newaxis]
            err = gradient - grad_exact
            err = err - err.mean(axis=1)[..., np.newaxis] # for simplex or exp update mean doesn't matter

            out_log = {'epoch': k,
                           'suboptimality': sub,
                           'learning_rate': lrk,
                           'pi_entropy': np.mean(entropy(self.pi.probas, axis=1))
                           }
            
            if dk is not None:
                out_log['divergence'] = dk
            
            if not exact:
                out_log['normalized_grad_error'] = np.max(np.abs(err)) / np.max(np.abs(grad_exact))
                out_log['samples'] = self.env.samples

            info.append(out_log)
            
            if verbose:
                print(f'Epoch: {k}, suboptimality: {sub}')

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

        old_state, info = self.env.reset(samples=False)
        if state is not None:
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
    

    def estimate_value(self):
        Q = np.zeros_like(self.env.R)

        for s in range(self.env.n_obs):
            for a in range(self.env.n_act):
                for _ in range(self.n_trajectories):
                    states, actions, rewards = self.sample_trajectory(s, a)
                    rtg = self.reward_to_go(rewards)
                    Q[s, a] += rtg[0]

        Q = Q / self.n_trajectories
        return Q
    

    def grad(self, Q):
        return self.grad_exact(Q)

    def grad_exact(self, Q):
        raise NotImplementedError

    def lrk(self, lr, k, grad):
        return lr, None


class PolicyIteration(Algorithm):
    def __init__(self, env: Tabular, horizon: int, n_trajectories: int):
        pi = DirectPolicy(env.observation_space, env.action_space)
        super().__init__(env, horizon, n_trajectories, pi, 'PI')


    def grad_exact(self, Q):
        return Q


    def lrk(self, lr, k, gradient):
        return np.inf, None

        
class REINFORCE(Algorithm):
    # Projected Policy Gradient Ascent using REINFORCE
    def __init__(self, env: Tabular, horizon: int, n_trajectories: int):
        pi = DirectPolicy(env.observation_space, env.action_space)
        super().__init__(env, horizon, pi, 'REINFORCE-PPG')


class PQA(Algorithm):
    # Projected Q-Ascent using MC rollouts to estimate Q
    def __init__(self, env: Tabular, horizon: int, n_trajectories: int):
        pi = DirectPolicy(env.observation_space, env.action_space)
        super().__init__(env, horizon, n_trajectories, pi, 'PQA')

    
    def grad_exact(self, Q):
        return Q
    
    
    def lrk(self, lr, k, gradient):
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = np.max(np.linalg.norm(greedy - pk, axis=1)**2 / 2)
        if dk == 0: # CONVERGED! V* = TV*
            return 0

        lrk = lr(k) * dk if callable(lr) else lr * dk / self.gamma**(2*(k + 1))
        return lrk, dk
    

class EQA(Algorithm):
    # Exponentiated Q-Ascent (NPG) using MC rollouts to estimate Q
    def __init__(self, env: Tabular, horizon: int, n_trajectories: int):
        pi = SoftmaxPolicy(env.observation_space, env.action_space)
        super().__init__(env, horizon, n_trajectories, pi, 'EQA')

    
    def grad_exact(self, Q):
        return Q


    def lrk(self, lr, k, gradient):
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = np.max(entropy(greedy, pk, axis=1))
        if dk == 0: # CONVERGED! V* = TV*
            return 0

        lrk = dk * lr(k) if callable(lr) else lr * dk / self.gamma**(2*(k + 1))
        return lrk, dk


class DPMD(Algorithm):
    # Exponentiated Q-Ascent (NPG) using d-step lookahead to estimate Q
    def __init__(self, pi: Policy, name: str, env: Tabular, horizon: int=None,
                 M_l: int=None, d: int=1, M_t: int = None, M_f: int = None):
        
        M_t = M_t or M_l
        M_f = M_f or M_l
        
        self.M_l = M_l
        self.M_t = M_t
        self.M_f = M_f
        self.d = d
        super().__init__(env, horizon, M_l, pi, name)

    
    def grad(self, V):
        tdm = V
        for _ in range(self.d - 1):
            tdm = self.approx_bellman(tdm)
        
        Q_d = np.zeros_like(self.env.R)
        for s in range(self.env.n_obs):
            for a in range(self.env.n_act):
                for _ in range(self.M_f):
                    self.env.state = s
                    sp, r, _ = self.env.step(a)
                    Q_d[s, a] += r + self.gamma * tdm[sp]
                        
        Q_d /= self.M_f
        return Q_d


    def grad_exact(self, Q):
        V = np.sum(self.pi.probas * Q, axis=1)
        tdm = V

        for _ in range(self.d - 1):
            tdm = self.env.bellman(tdm)

        next_Vd = self.env.P.transpose((2, 1, 0)) @ tdm
        Qd = self.env.R + self.gamma * next_Vd.transpose((1, 0))
        return Qd


    def lrk(self, lr, k, gradient):
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = self.divergence(greedy, pk)
        if dk == 0: # CONVERGED! V* = TV*
            return 0

        lrk = lr(k) * dk if callable(lr) else lr * dk / self.gamma**(2*(k + 1) * self.d)
        if np.isinf(lrk) or np.isnan(lrk):
            lrk = np.inf
        return lrk, dk
    

    def divergence(self, u, v):
        raise NotImplementedError
    

    def estimate_value(self):
        V_l = np.zeros((self.env.observation_space.n, ))
        for s in range(self.env.n_obs):
            for t in range(self.M_l):
                states, actions, rewards = self.sample_trajectory(s)
                rtg = self.reward_to_go(rewards)
                V_l[s] += rtg[0]
            
            V_l[s] /= self.M_l
        
        return V_l


    def approx_bellman(self, V):
        Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        for s in range(self.env.n_obs):
            for a in range(self.env.n_act):
                for t in range(self.M_t):

                    self.env.state = s
                    sp, r, _ = self.env.step(a)
                    Q[s, a] = r + self.gamma * V[sp]

        return np.max(Q, axis=1)
    

    def diagnostics(self):
        pass


class DEQA(DPMD):
    def __init__(self, env: Tabular, horizon: int = None, M_l: int = None, d: int = 1, M_t: int = None, M_f: int = None):
        pi = SoftmaxPolicy(env.observation_space, env.action_space)
        name = f'{d}-EQA'
        super().__init__(pi, name, env, horizon, M_l, d, M_t, M_f)


    def divergence(self, u, v): # KL-divergence
        return np.max(entropy(u, v, axis=1))


class DPQA(DPMD):
    def __init__(self, env: Tabular, horizon: int, M_l: int, d: int = 1, M_t: int = None, M_f: int = None):
        pi = DirectPolicy(env.observation_space, env.action_space)
        name = f'{d}-PQA'
        super().__init__(pi, name, env, horizon, M_l, d, M_t, M_f)


    def divergence(self, u, v): # Total variation divergence
        return np.max(np.linalg.norm(u - v, axis=1)**2 / 2)
