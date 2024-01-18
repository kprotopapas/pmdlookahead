import numpy as np
import time

from environments.gymnax import JDeepsea
from environments.solvable import Deepsea

size = 64
gamma = 0.99
jenv = JDeepsea(size, gamma)

start = time.time()
JQ = jenv.optimal()
end = time.time()


duration = end - start
print(f'PI took {end - start} seconds')

Q = np.array(JQ)
np.save(f'backups/Deepsea-{size}-0.99.npy', Q, allow_pickle=True)
