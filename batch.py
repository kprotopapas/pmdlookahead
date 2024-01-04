import os
import subprocess
import argparse
import time
from helper import parse_args

if __name__ == '__main__':
    args = parse_args(file=True)

    file = args.file

    size = args.size
    horizon = args.horizon

    alg = args.alg.upper()
    n_agents = args.n_agents
    lr = args.lr
    epochs = args.epochs
    d = args.d
    nsims = args.simulations
    simstring = '' if nsims is None else f'--simulations {nsims}'
    
    group = args.group

    for i in range(n_agents):
        if f'{group}_{d}' not in os.listdir('runs'):
            os.mkdir(f'runs/{group}_{d}')
        
        os.mkdir(f'runs/{group}_{d}/{i}')

        script = \
        f"""#!/bin/bash
#SBATCH -n 4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=20:00
#SBATCH --job-name={group}/{d}/{i}
#SBATCH --output=runs/{group}_{d}/{i}/output.out
#SBATCH --error=runs/{group}_{d}/{i}/error.err

module load gcc/8.2.0 python/3.10.4
source cluster-venv/bin/activate
wandb offline
python {file}.py --alg {alg} -n 1 -s {size} -t {horizon} --epochs {epochs} --wandb --group {group} -d {d} {simstring}
"""
        
        with open(f'script_{d}_{i}', 'w') as f:
            f.write(script)
        
        subprocess.run(['sbatch', f'script_{d}_{i}'])

        time.sleep(0.1)
        os.remove(f'script_{d}_{i}')
