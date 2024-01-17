from typing import Callable
import chex
from gymnax import EnvParams, EnvState
import wandb
from jax import numpy as jnp
import numpy as np
import logging
import time
import jax
from functools import partial

from scipy.stats import entropy
from gymnax.environments.environment import Environment
from algorithms.policies import JPolicy, JSoftmaxPolicy
from environments.gymnax.tabular import JTabular


class JAlgorithm(object):
    def __init__(self, env: JTabular, horizon: int, n_trajectories: int, pi: JPolicy, name: str):
        self.config = {'algorithm': name,
                       'environmetn': {'name': env.name, 'gamma': env.gamma},
                       'horizon': horizon,
                       'trajectories': n_trajectories}
        self.horizon = horizon
        self.n_trajectories = n_trajectories
        self.pi = pi
        self.name = name
        self.env = env


    def learn(self, epochs: int = 1000, lr: Callable | float = 1,
              exact=False, use_wandb=True, verbose=False, group=None, seed=42):

        self.exact = exact
        self.seed = seed
        env = self.env
        params = env.default_params
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
                logging.info(f'Epoch: {k}, reward: {reward:06f},\
                             in time: {duration:06f}, improvement: {improvement:06f}')

            if use_wandb:
                wandb.log(metrics)
                

        if use_wandb:
            if verbose:
                print('Finishing wandb job...')
            run.finish()

        return info


    def reward_to_go(self, rewards: chex.Array):
        discounts = jnp.power(self.env.gamma, jnp.arange(len(rewards)))
        discounted = rewards * discounts
        rtg = jnp.cumsum(discounted[::-1])[::-1] / discounts
        return rtg
    
    
    @partial(jax.jit, static_argnums=(0, 2))
    def sample_trajectory(self, key: chex.PRNGKey, env: JTabular, params: EnvParams):
        env_key, action_key = jax.random.split(key)
        obs, state = env.reset(env_key, params)
        priors = jnp.zeros((self.horizon, env.num_actions))

        return self.sample_trajectory_from_state(action_key, env, params, obs, state, priors)
    
    
    @partial(jax.jit, static_argnums=(0, 2))
    def sample_trajectory_from_state(self, key: chex.PRNGKey, env: JTabular, params: EnvParams,
                          obs: chex.Array, state: EnvState, priors: chex.Array):
        
        def step(a: tuple[chex.PRNGKey, chex.Array, EnvState, JPolicy, jnp.bool_], prior: chex.Array):
            key, obs, state, pi, old_done = a
            key, action_key = jax.random.split(key)
            logits = pi.act(obs)
            action = jax.random.categorical(action_key, logits + prior)
            nobs, nstate, reward, done, _ = env.step(key, state, action, params)
            reward = jnp.where(old_done, 0, reward)
            return (key, nobs, nstate, pi, done | old_done), (obs, action, reward)


        carry, ys = jax.lax.scan(step, (key, obs, state, self.pi, jnp.bool_(False)), priors, length=self.horizon)

        key, obs, state, _, done = carry
        observations, actions, rewards = ys
        observations = jnp.vstack(([obs, observations]))


        return observations, actions, rewards # TODO for more advanced, flag "done" states as invalid...


    @partial(jax.jit, static_argnums=(0, 2))
    def estimate_value(self, key: chex.PRNGKey, env: JTabular, params: EnvParams):
        def value(value_key, obs, state, priors):
            _, _, rewards = self.sample_trajectory_from_state(value_key, env, params, obs, state, priors)
            rtg = self.reward_to_go(rewards)
            return rtg[0]

        over_priors = jax.vmap(value, (None, None, None, 0))
        over_both = jax.vmap(over_priors, (None, 0, 0, None))
        over_keys = jax.vmap(over_both, (0, None, None, None))
        
        state_nums = jnp.arange(env.n_states)
        inverse = lambda s: env.inverse(s)
        obs_batch, state_batch = jax.vmap(inverse, (0,))(state_nums)

        priors_batch = jnp.zeros((env.num_actions, self.horizon, env.num_actions))
        for i in range(env.num_actions):
            priors_batch = priors_batch.at[i, 0, i].set(jnp.inf)

        keys = jax.random.split(key, self.n_trajectories)

        Q_all = over_keys(keys, obs_batch, state_batch, priors_batch)

        return jnp.mean(Q_all, axis=0)
    

    def lrk(self, gradient: chex.Array, k: int):
        raise NotImplementedError
    
    def grad(self, Q):
        return Q


class JPolicyIteration(JAlgorithm):
    def __init__(self, env: JTabular, horizon: int, n_trajectories: int, pi: JPolicy):
        super().__init__(env, horizon, n_trajectories, pi, 'PI')


    def lrk(self, k: int, gradient: chex.Array):
        return jnp.inf


class REINFORCE(JAlgorithm):
    # Projected Policy Gradient Ascent using REINFORCE
    def __init__(self, env: JTabular, horizon: int, n_trajectories: int, pi: JPolicy):
        super().__init__(env, horizon, n_trajectories, pi, 'REINFORCE')


class JPQA(JAlgorithm):
    # Projected Q-Ascent using MC rollouts to estimate Q
    def __init__(self, env: JTabular, horizon: int, n_trajectories: int):
        pi = JPolicy()
        super().__init__(env, horizon, n_trajectories, pi, 'PQA')
        

    def lrk(self, k: int, gradient: chex.Array):
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = np.max(np.linalg.norm(greedy - pk, axis=1)**2 / 2)
        if dk == 0: # CONVERGED! V* = TV*
            return 0, 0
        
        lrk = dk / self.env.gamma**(2*(k + 1))
        return lrk, dk
    

class JEQA(JAlgorithm):
    # Exponentiated Q-Ascent (NPG) using MC rollouts to estimate Q
    def __init__(self, env: JTabular, horizon: int, n_trajectories: int):
        pi = JSoftmaxPolicy.uniform(env.n_states, env.num_actions)
        super().__init__(env, horizon, n_trajectories, pi, 'EQA')


    def lrk(self, k: int, gradient: chex.Array):
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = np.max(entropy(greedy, pk, axis=1))
        if dk == 0: # CONVERGED! V* = TV*
            return 0, 0
        
        lrk = dk / self.env.gamma**(2*(k + 1))
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
                    Q[s, a] = r + self.env.gamma * V[sp]

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