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
import os

from scipy.stats import entropy
from gymnax.environments.environment import Environment
from algorithms.policies import JDirectPolicy, JPolicy, JSoftmaxPolicy
from environments.gymnax.tabular import JTabular

logging.basicConfig(level=logging.INFO)


class JAlgorithm(object):
    def __init__(self, env: JTabular, horizon: int, n_trajectories: int, pi: JPolicy, name: str):
        self.config = {'algorithm': name,
                       'environment': {'name': env.name, 'gamma': env.gamma},
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

        backup = f'{env.display_name}-{env.size}-{env.gamma}.npy'
        if backup in os.listdir('backups'):
            optimal = jnp.load(f'backups/{backup}', allow_pickle=True)
        else:
            optimal = env.optimal()
        
        V_opt = optimal.max(axis=1)
        V = env.V(self.pi)
        old = jnp.dot(self.env.mu, V_opt - V)


        for k in range(epochs):
            key = jax.random.PRNGKey(seed * k)
            start = time.time()
            
            Q = self.env.Q(self.pi)
            
            value_key, grad_key = jax.random.split(key, 2)
            grad_exact = self.grad_exact(key, env, params, Q)
            Q_est = self.estimate_value(value_key, env, params, self.pi)
            grad = self.grad(grad_key, env, params, Q_est)

            gradient = jnp.where(exact, grad_exact, grad)

            lk, dk = self.lrk(lr, k, gradient)
            new_pi = self.pi.update(gradient, lk)
            if jnp.any(jnp.isnan(new_pi.probas)):
                pass
            self.pi = new_pi


            V = jnp.sum(self.pi.probas * Q, axis=1)
            
            sub = float(jnp.dot(self.env.mu, V_opt - V))
            right_probability = jnp.mean(self.pi.probas[:, 1])
            metrics = {'suboptimality': sub, 'epoch': k, 'divergence': dk, 'learning_rate': lk,
                       'right_probability': right_probability}
            
            if not exact:
                metrics['grad_error'] = jnp.max(jnp.abs(grad - grad_exact))
                greedy_exact = jnp.argmax(grad_exact, axis=1)
                greedy = jnp.argmax(grad, axis=1)
                count = jnp.count_nonzero(jnp.logical_xor(greedy_exact, greedy))
                metrics['diff_greedy'] = count

            duration = time.time() - start
            info.append(metrics)

            improvement = old - sub
            old = sub
            sub = jnp.dot(self.env.mu, V_opt - V)

            if verbose:
                logging.info(f'Epoch: {k}, suboptimality: {sub:06f}, in time: {duration:06f}, improvement: {improvement:06f}')

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
    def sample_trajectory(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, pi: JPolicy):
        env_key, action_key = jax.random.split(key)
        obs, state = env.reset(env_key, params)
        priors = jnp.zeros((self.horizon, env.num_actions))

        return self.sample_trajectory_from_state(action_key, env, params, obs, state, priors, pi)
    
    
    @partial(jax.jit, static_argnums=(0, 2))
    def sample_trajectory_from_state(self, key: chex.PRNGKey, env: JTabular, params: EnvParams,
                          obs: chex.Array, state: EnvState, priors: chex.Array, pi: JPolicy):
        
        def step(a: tuple[chex.PRNGKey, chex.Array, EnvState, jnp.bool_], prior: chex.Array):
            key, obs, state, old_done = a
            nkey, action_key = jax.random.split(key)
            logits = pi.act(obs)
            action = jax.random.categorical(action_key, logits + prior)
            nobs, nstate, reward, done, _ = env.step(nkey, state, action, params)
            reward = jnp.where(old_done, 0, reward)
            return (nkey, nobs, nstate, done | old_done), (obs, action, reward)


        carry, ys = jax.lax.scan(step, (key, obs, state, jnp.bool_(False)), priors, length=self.horizon)

        nkey, nobs, nstate, done = carry
        observations, actions, rewards = ys
        observations = jnp.vstack(([obs, observations]))


        return observations, actions, rewards # TODO for more advanced, flag "done" states as invalid...
    
    @partial(jax.jit, static_argnums=(0, 2))
    def debug(self, key: chex.PRNGKey, env: JTabular, params: EnvParams,
                          obs: chex.Array, state: EnvState, priors: chex.Array, pi: JPolicy):
        
        def step(a: tuple[chex.PRNGKey, chex.Array, EnvState, jnp.bool_], prior: chex.Array):
            key, obs, state, old_done = a
            nkey, action_key = jax.random.split(key)
            logits = pi.act(obs)
            belief = logits + prior
            action = jax.random.categorical(action_key, belief)
            nobs, nstate, reward, done, _ = env.step(nkey, state, action, params)
            reward = jnp.where(old_done, 0, reward)
            return (nkey, nobs, nstate, done | old_done), (obs, action, reward, belief)


        carry, ys = jax.lax.scan(step, (key, obs, state, jnp.bool_(False)), priors, length=self.horizon)

        nkey, nobs, nstate, done = carry
        observations, actions, rewards, beliefs = ys
        observations = jnp.vstack(([obs, observations]))


        return observations, actions, rewards, beliefs # TODO for more advanced, flag "done" states as invalid...


    @partial(jax.jit, static_argnums=(0, 2))
    def estimate_value(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, pi: JPolicy):
        def value(value_key, obs, state, priors):
            _, _, rewards = self.sample_trajectory_from_state(value_key, env, params, obs, state, priors, pi)
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
    

    @partial(jax.jit, static_argnums=(0, 2, 4))
    def evaluate(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, rollouts: int) -> float:
        keys = jax.random.split(key, rollouts)

        def value(key, env, params):
            _, _, rewards = self.sample_trajectory(key, env, params)
            r = self.reward_to_go(rewards)[0]
            return r
        
        rewards = jax.vmap(value, (0, None, None))(keys, env, params)
        return jnp.mean(rewards)
    

    def lrk(self, lr, k: int, gradient: chex.Array):
        raise NotImplementedError
    
    @partial(jax.jit, static_argnums=(0, 2))
    def grad(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, Q: jnp.ndarray):
        return self.grad_exact(key, env, params, Q)
    
    @partial(jax.jit, static_argnums=(0, 2))
    def grad_exact(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, Q: jnp.ndarray):
        return Q


class JPolicyIteration(JAlgorithm):
    def __init__(self, env: JTabular, horizon: int, n_trajectories: int, pi: JPolicy):
        super().__init__(env, horizon, n_trajectories, pi, 'JPI')


    def lrk(self, lr, k: int, gradient: chex.Array):
        return jnp.inf, 0


class JREINFORCE(JAlgorithm):
    # Projected Policy Gradient Ascent using REINFORCE
    def __init__(self, env: JTabular, horizon: int, n_trajectories: int, pi: JPolicy):
        super().__init__(env, horizon, n_trajectories, pi, 'JREINFORCE')


class JPQA(JAlgorithm):
    # Projected Q-Ascent using MC rollouts to estimate Q
    def __init__(self, env: JTabular, horizon: int, n_trajectories: int):
        pi = JDirectPolicy.uniform(env.n_states, env.num_actions)
        super().__init__(env, horizon, n_trajectories, pi, 'JPQA')
        

    def lrk(self, lr: Callable | float, k: int, gradient: chex.Array):
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = np.max(np.linalg.norm(greedy - pk, axis=1)**2 / 2)
        if dk == 0: # CONVERGED! V* = TV*
            return 0, 0
        
        lrk = dk / self.env.gamma**(2*(k + 1))
        if callable(lr):
            return lr(k), dk
        return lrk, dk
    

class JEQA(JAlgorithm):
    # Exponentiated Q-Ascent (NPG) using MC rollouts to estimate Q
    def __init__(self, env: JTabular, horizon: int, n_trajectories: int):
        pi = JSoftmaxPolicy.uniform(env.n_states, env.num_actions)
        super().__init__(env, horizon, n_trajectories, pi, 'JEQA')


    def lrk(self, lr: Callable | float, k: int, gradient: chex.Array):
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = np.max(entropy(greedy, pk, axis=1))
        if dk == 0: # CONVERGED! V* = TV*
            return 0, 0
        
        lrk = dk / self.env.gamma**(2*(k + 1))
        if callable(lr):
            return lr(k), dk
        return lrk, dk


class JDPMD(JAlgorithm):
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

    
    @partial(jax.jit, static_argnums=(0, 2))
    def grad(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, V: jnp.ndarray):
        body = lambda _, y: self.approx_bellman(key, env, params, y)
        
        tdm = jax.lax.fori_loop(0, self.d - 1, body, V)

        def value(key, state, action):
            nobs, nstate, reward, done, _ = env.step(key, state, action, params)
            nind = env.index(nobs)
            return reward + self.env.gamma * tdm[nind]
        
        state_nums = jnp.arange(env.n_states)
        inverse = lambda s: env.inverse(s)
        obs_batch, state_batch = jax.vmap(inverse, (0,))(state_nums)
        action_batch = jnp.arange(env.num_actions)
        keys = jax.random.split(key, self.M_t)
        
        over_states = jax.vmap(value, (None, 0, None), 0)
        over_actions = jax.vmap(over_states, (None, None, 0), 1)
        over_keys = jax.vmap(over_actions, (0, None, None), 0)

        Q = over_keys(keys, state_batch, action_batch).mean(axis=0)

        return Q



    @partial(jax.jit, static_argnums=(0, 2))
    def grad_exact(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, Q: jnp.ndarray, pi: JPolicy):
        V = np.sum(pi.probas * Q, axis=1)

        body = lambda _, y: env.bellman(y)
        tdm = jax.lax.fori_loop(0, self.d - 1, body, V)

        next_Vd = jnp.matmul(self.env.P.transpose((2, 1, 0)), tdm)
        Qd = self.env.R + self.env.gamma * next_Vd.transpose((1, 0))
        return Qd


    def lrk(self, lr, k, gradient):
        greedy = 1 * (gradient == gradient.max(axis=1)[..., np.newaxis])
        greedy = greedy / greedy.sum(axis=1)[..., np.newaxis]
        pk = self.pi.probas

        dk = self.divergence(greedy, pk)
        if dk == 0: # CONVERGED! V* = TV*
            return 0, 0

        lrk = lr(k) if callable(lr) else lr * dk / self.env.gamma**(2*(k + 1) * self.d) # TODO add adaptive flag
        if np.isinf(lrk) or np.isnan(lrk):
            lrk = np.inf
        return lrk, dk
    

    def divergence(self, u, v):
        raise NotImplementedError
    
    
    @partial(jax.jit, static_argnums=(0, 2))
    def estimate_value(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, pi: JPolicy) -> chex.Array:
        priors = jnp.zeros((self.horizon, env.num_actions))

        def value(value_key, obs, state):
            _, _, rewards = self.sample_trajectory_from_state(value_key, env, params, obs, state, priors, pi)
            rtg = self.reward_to_go(rewards)
            return rtg[0]

        over_states = jax.vmap(value, (None, 0, 0))
        over_keys = jax.vmap(over_states, (0, None, None))
        
        state_nums = jnp.arange(env.n_states)
        inverse = lambda s: env.inverse(s)
        obs_batch, state_batch = jax.vmap(inverse, (0,))(state_nums)

        keys = jax.random.split(key, self.n_trajectories)
        V_all = over_keys(keys, obs_batch, state_batch)

        return jnp.mean(V_all, axis=0)

    

    @partial(jax.jit, static_argnums=(0, 2))
    def approx_bellman(self, key: chex.PRNGKey, env: JTabular, params: EnvParams, V: jnp.ndarray):
        state_nums = jnp.arange(env.n_states)

        def value(key, state, action):
            nobs, nstate, reward, done, _ = env.step(key, state, action, params)
            nind = env.index(nobs)
            return reward + self.env.gamma * V[nind]
        
        inverse = lambda s: env.inverse(s)
        obs_batch, state_batch = jax.vmap(inverse, (0,))(state_nums)
        action_batch = jnp.arange(env.num_actions)
        keys = jax.random.split(key, self.M_t)
        
        over_states = jax.vmap(value, (None, 0, None), 0)
        over_actions = jax.vmap(over_states, (None, None, 0), 1)
        over_keys = jax.vmap(over_actions, (0, None, None), 0)

        Q = over_keys(keys, state_batch, action_batch).mean(axis=0)

        return np.max(Q, axis=1)
    


class JDEQA(JDPMD):
    def __init__(self, env: JTabular, horizon: int = None, M_l: int = None, d: int = 1, M_t: int = None, M_f: int = None):
        pi = JSoftmaxPolicy.uniform(env.n_states, env.num_actions)
        name = f'{d}-JEQA'
        super().__init__(pi, name, env, horizon, M_l, d, M_t, M_f)


    def divergence(self, u, v): # KL-divergence
        return np.max(entropy(u, v, axis=1))


class JDPQA(JDPMD):
    def __init__(self, env: JTabular, horizon: int, M_l: int = None, d: int = 1, M_t: int = None, M_f: int = None):
        pi = JDirectPolicy.uniform(env.n_states, env.num_actions)
        name = f'{d}-JPQA'
        super().__init__(pi, name, env, horizon, M_l, d, M_t, M_f)


    def divergence(self, u, v): # Total variation divergence
        return jnp.max(jnp.linalg.norm(u - v, axis=1)**2 / 2)