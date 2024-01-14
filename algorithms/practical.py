from typing import Callable
import chex
from gymnax import EnvParams, EnvState
import wandb
from jax import numpy as jnp
import numpy as np
import logging
import time
import jax

from scipy.stats import entropy
from gymnax.environments.environment import Environment
from algorithms.policies import JPolicy, JSoftmaxPolicy
from environments.jym.tabular import JTabular


class JAlgorithm(object):
    def __init__(self, gamma: float, horizon: int, n_trajectories: int, pi: JPolicy, name: str):
        self.config = {'algorithm': name,
                       'gamma': gamma,
                       'horizon': horizon}
        self.horizon = horizon
        self.n_trajectories = n_trajectories
        self.gamma = gamma
        self.pi = pi


    def learn(self, env: JTabular, params: EnvParams, epochs: int = 1000, lr: Callable | float = 1,
              exact=False, use_wandb=True, verbose=False, group=None, seed=None):

        self.exact = exact
        self.env = env
        self.seed = seed

        info = []
        if use_wandb:
            config = {
                    'learning_rate': 'custom' if callable(lr) else lr,
                    'epochs': epochs,
                    'exact': exact
                }
            for key, value in self.config.items():
                config[key] = value
        
            run = wandb.init(project="rl_experiments", config=config, group=group, entity='kprotopapas')

        old = 0

        ostart, sstart = self.env.reset(key)
        for k in range(epochs):
            key = jax.random.PRNGKey(seed * k)
            start = time.time()
            
            Q = self.estimate_value(key, env, params)
            gradient = self.grad(Q)
            lr = self.lrk(Q, k)
            self.pi = self.pi.update(gradient, lr)

            reward = jnp.max(jnp.abs(Q))
            metrics = {'reward': reward}

            duration = time.time() - start
            info.append(metrics)

            improvement = reward - old
            old = reward

            if verbose:
                logging.info(f'Epoch: {k}, reward: {reward:06f},
                             in time: {duration:06f}, improvement: {improvement:06f}')

            if use_wandb:
                wandb.log(metrics)
                

        if use_wandb:
            if verbose:
                print('Finishing wandb job...')
            run.finish()

        return info


    def reward_to_go(self, rewards: chex.Array):
        discounts = jnp.power(self.gamma, jnp.arange(len(rewards)))
        discounted = rewards * discounts
        rtg = jnp.cumsum(discounted[::-1])[::-1] / discounts
        return rtg


    def sample_trajectory(self, key: chex.PRNGKey, env: Environment, params: EnvParams,
                          state: EnvState = None, obs: chex.Array = None, action: int = None):
        rewards = jnp.ndarray((self.horizon, ))
        observations = jnp.ndarray((self.horizon + 1,))
        actions = jnp.ndarray((self.horizon,))

        reset_key, action_key = jax.random.split(key)

        obs, state = env.reset(reset_key, params)
        observations = observations.at[0].set(obs)

        def cond(a):
            t, done, _, _ = a
            return done | t == self.horizon

        def step(a):
            t, _, obs, state = a
            action = self.pi.act(obs)
            obs, state, reward, done, _ = env.step(action_key, state, action, params)
            rewards.at[t].set(reward)
            actions.at[t].set(action)
            observations.at[t + 1].set(obs)

            return t + 1, done, obs, state

        _ = jax.lax.while_loop(cond, step, (0, False, obs, state))

        return observations, actions, rewards
    
    
    def fit(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, k: int):
        raise NotImplementedError

    def estimate_value(self, key: chex.PRNGKey, env: Environment, params: EnvParams):
        Q = jnp.array((env.n_states, env.num_actions))

        for s in range(env.n_states):
            for a in range(env.n_actions):
                for _ in range(self.n_trajectories):
                    observations, actions, rewards = self.sample_trajectory(key, env, params, s, s, a)
                    rtg = self.reward_to_go(rewards)
                    Q = Q.at[s, a].add(rtg[0])
            

        Q = Q / self.n_trajectories
    

    def lrk(self, k: int, gradient):
        raise NotImplementedError
    
    def grad(self, Q):
        raise NotImplementedError


class PolicyIteration(JAlgorithm):
    def __init__(self, gamma: float, horizon: int, n_trajectories: int, pi: JPolicy):
        super().__init__(gamma, horizon, n_trajectories, pi, 'PI')


    def fit(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, k: int):
        Q = self.estimate_value(key, env, params)

        self.pi = self.pi.update(Q, jnp.inf)

        return Q, {}
        
        
class REINFORCE(JAlgorithm):
    # Projected Policy Gradient Ascent using REINFORCE
    def __init__(self, gamma: float, horizon: int, n_trajectories: int, pi: JPolicy):
        super().__init__(gamma, horizon, n_trajectories, pi, 'REINFORCE')


class PQA(JAlgorithm):
    # Projected Q-Ascent using MC rollouts to estimate Q
    def __init__(self, gamma: float, horizon: int, n_trajectories: int, pi: JPolicy):
        super().__init__(gamma, horizon, n_trajectories, pi, 'PQA')
        

    def fit(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, k: int):
        states, actions, rewards = self.sample_trajectory(key, env, params)


    
    def lrk(self, lr, k, gradient):
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = np.max(np.linalg.norm(greedy - pk, axis=1)**2 / 2)
        if dk == 0: # CONVERGED! V* = TV*
            return 0, 0

        lrk = lr(k) * dk if callable(lr) else lr * dk / self.gamma**(2*(k + 1))
        return lrk, dk
    

class EQA(JAlgorithm):
    # Exponentiated Q-Ascent (NPG) using MC rollouts to estimate Q
    def __init__(self, env: JTabular, horizon: int, n_trajectories: int):
        pi = JSoftmaxPolicy(env.observation_space.n, env.action_space.n)
        super().__init__(env, horizon, n_trajectories, pi, 'EQA')


    def estimate_value(self):
        Q = np.zeros_like(self.env.R)
        self.env.reset()
        states, actions, rewards = self.sample_trajectory()
        rtg = self.env.rtg(rewards)
        for s, a, r in zip(states, actions, rtg):
            Q[s, a] += r # or maybe average???

        return Q

    
    def grad_exact(self, Q):
        return Q


    def lrk(self, lr, k, gradient):
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = np.max(entropy(greedy, pk, axis=1))
        if dk == 0: # CONVERGED! V* = TV*
            return 0, 0

        lrk = dk * lr(k) if callable(lr) else lr * dk / self.gamma**(2*(k + 1))
        return lrk, dk


class DPMD(JAlgorithm):
    # Exponentiated Q-Ascent (NPG) using d-step lookahead to estimate Q
    def __init__(self, pi: JPolicy, name: str, env: JTabular, horizon: int = None,
                 M_l: int=None, d: int=1, M_t: int = None, M_f: int = None):
        
        M_t = M_t or M_l
        M_f = M_f or M_l
        
        self.M_l = M_l
        self.M_t = M_t
        self.M_f = M_f
        self.d = d
        super().__init__(env, horizon, M_l, pi, name)
        self.config['n_tree'] = M_t
        self.config['n_final'] = M_f

    
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
            return 0, 0

        lrk = lr(k) * dk if callable(lr) else lr * dk / self.gamma**(2*(k + 1) * self.d) #TODO MAKE adaptive
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


class DEQA(DPMD):
    def __init__(self, env: JTabular, horizon: int = None, M_l: int = None, d: int = 1, M_t: int = None, M_f: int = None):
        pi = JSoftmaxPolicy(env.observation_space.n, env.action_space.n)
        name = f'{d}-EQA'
        super().__init__(pi, name, env, horizon, M_l, d, M_t, M_f)


    def divergence(self, u, v): # KL-divergence
        return np.max(entropy(u, v, axis=1))


class DPQA(DPMD):
    def __init__(self, env: JTabular, horizon: int, M_l: int = None, d: int = 1, M_t: int = None, M_f: int = None):
        pi = JSoftmaxPolicy(env.observation_space, env.action_space)
        name = f'{d}-PQA'
        super().__init__(pi, name, env, horizon, M_l, d, M_t, M_f)


    def divergence(self, u, v): # Total variation divergence
        return np.max(np.linalg.norm(u - v, axis=1)**2 / 2)