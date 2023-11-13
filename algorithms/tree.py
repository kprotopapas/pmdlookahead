from algorithms.baselines import DPMD
from algorithms.policies import SoftmaxPolicy
from environments.discrete import Tabular


class TSPMD(DPMD):
    def __init__(self, env: Tabular, d=2, horizon=10, n_trajectories=10):
        pi = SoftmaxPolicy(env.n_obs, env.n_act)
        super().__init__(env, horizon, n_trajectories, pi, f'{d}-TSPMD')


    def sample_trajectory(self, state=None, action=None):
        return super().sample_trajectory(state, action)