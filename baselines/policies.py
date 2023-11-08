import numpy as np

from helper import *



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

    
    @property
    def probas(self):
        return self.probabilities


class SoftmaxPolicy(Policy):
    def __init__(self, states, actions):
        n_states = states.n
        n_actions = actions.n

        self.n_states = n_states
        self.n_actions = n_actions
        self.params = np.zeros((n_states, n_actions))
        self.actions = actions
    

    def sample(self, state):
        action = np.random.choice(np.arange(self.n_actions), p=self.probas[state])
        return action


    def update(self, grad, lr: float):
        if lr == np.inf:
            self.params = np.where(grad == grad.max(axis=1)[..., np.newaxis], np.inf, -np.inf)
            return

        grad = grad - grad.mean(axis=1)[..., np.newaxis] # center gradient before update
        new = self.params + lr * grad
        new_centered = new - new.max(axis=1)[..., np.newaxis] # renormalize by max
        if np.any(np.isnan(new_centered)):
            pass
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