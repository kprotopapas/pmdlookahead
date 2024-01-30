import timeit
import numpy as np
import time
from algorithms.jym import *
from algorithms.policies import *
from algorithms.tree import PGMCTS
from jax import numpy as jnp
import jax
import gymnax

from helper import *

from environments.jym import DeepSea, Catch, JDeepsea, Gridworld
from environments.solvable import Deepsea

env = gymnax.make('Breakout-MinAtar')