from __future__ import annotations

import chex
from typing import Callable
import numpy as np
from jax import numpy as jnp

from helper import *


@chex.dataclass(kw_only=True)
class JPolicy(object):
    embedding = jnp.ones((1,))
    # Returns logits associated with current 
    def act(self, obs: chex.Array) -> chex.Array:
        raise NotImplementedError

    def update(self, grad: chex.Array, lr: float) -> JPolicy:
        raise NotImplementedError


class Policy:
    def sample(self, state):
        raise NotImplementedError
    
    def logprob(self, state, actions):
        raise NotImplementedError
    
    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
    
    def update(self, grad):
        raise NotImplementedError
    
    @property
    def probas(self):
        raise NotImplementedError
    

    def jax(self, embedding = lambda x: x) -> JPolicy:
        pass


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
        if np.isinf(lr):
            probas = 1. * (grad == grad.max(axis=1)[..., np.newaxis])
            probas /= probas.sum(axis=1)[..., np.newaxis]
            self.probabilities = probas
            return
        
        new = self.probabilities + lr * grad
        for i in range(self.probabilities.shape[0]):
            projected = simplex_project_sort(new[i] - new[i].mean()) # easier projection, take away (1, 1, 1, ...)
            self.probabilities[i] = projected

    
    def save(self, path):
        with open(path, 'wb') as f:
            np.save(f, self.probabilities, allow_pickle=True)

    
    def load(self, path):
        with open(path, 'rb') as f:
            self.probabilities = np.load(f, allow_pickle=True)

    
    @property
    def probas(self):
        return self.probabilities


@chex.dataclass(kw_only=True)
class JSoftmaxPolicy(JPolicy): # softmax policy
    logits: chex.Array

    def act(self, obs: chex.Array) -> chex.Array:
        state = jnp.sum(obs * self.embedding).astype(jnp.int32)
        return self.logits[state]


    def update(self, grad: chex.Array, ind: chex.Array, lr: float) -> JPolicy:
        grad = grad - grad.mean() #  center gradient before update
        
        new = self.logits[ind] + lr * grad
        new_centered = new - new.max() # renormalize by max
        new_logits = logits.at[ind].set(new_centered)

        argmax = jnp.where(grad == grad.max(), jnp.inf, -jnp.inf)
        logits = jnp.where(jnp.isinf(lr), argmax, new_logits)

        return JSoftmaxPolicy(embedding=self.embedding,
                                  logits=logits)


    def numpy(self):
        return SoftmaxPolicy(self.logits.shape[0], self.logits.shape[1])


    @staticmethod
    def uniform(n_states: int, n_actions: int, embedding = jnp.ones((1,))) -> JSoftmaxPolicy:
        return JSoftmaxPolicy(logits=jnp.zeros((n_states, n_actions)), embedding=embedding)


class SoftmaxPolicy(Policy):
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.params = np.zeros((n_states, n_actions))
    

    def sample(self, state):
        action = np.random.choice(np.arange(self.n_actions), p=self.probas[state])
        return action


    def update(self, grad, lr: float):
        if np.isinf(lr):
            self.params = np.where(grad == grad.max(axis=1)[..., np.newaxis], np.inf, -np.inf)
            return

        grad = grad - grad.mean(axis=1)[..., np.newaxis] # center gradient before update
        new = self.params + lr * grad
        new_centered = new - new.max(axis=1)[..., np.newaxis] # renormalize by max
        self.params = new_centered

    
    def save(self, path):
        with open(path, 'wb') as f:
            np.save(f, self.params, allow_pickle=True)

    
    def load(self, path):
        with open(path, 'rb') as f:
            self.params = np.load(f, allow_pickle=True)

    @property
    def probas(self):
        exps = np.exp(self.params)
        probas = exps / np.sum(exps, axis=1).reshape(-1, 1)
        nans = 1. * np.isnan(probas)
        fill = nans / np.maximum(nans.sum(axis=1)[..., np.newaxis], np.ones_like(nans))
        probas = np.where(np.isnan(probas), fill, probas)
        return probas

    
    def jax(self, embedding=jnp.ones((1,))) -> JPolicy:
        logits = jnp.array(self.params)
        return JSoftmaxPolicy(logits=logits, embedding=embedding)