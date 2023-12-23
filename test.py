import jax
from jax import numpy as jnp
import numpy as np
import time

from environments.jym import *
from environments.solvable import *
from algorithms.policies import *
from algorithms.tree import PGMCTS
from planning.jmcts import *
from helper import *

import wandb

args = parse_args()
group = args.group
depth = args.d
N = args.epochs

size = args.size
depth = args.d

gamma = 0.99
env = Deepsea(size, gamma)
jenv = JDeepsea(size=size, gamma=gamma)

embedding = lambda state: state[0] * size + state[1]
jpolicy = JSoftmaxPolicy.uniform(env.n_obs, env.n_act, embedding=embedding)

pgmcts = PGMCTS(policy=jpolicy, depth=depth, n_sims=500, M_l=16)


def callback(jpolicy: JSoftmaxPolicy):
    policy = SoftmaxPolicy(env.n_obs, env.n_act)
    logits = np.array(jpolicy.logits)
    policy.update(logits, 1)

    V = env.V(policy)

    print(V[env.state])


pgmcts.learn(jenv, 0.1, callback=callback)