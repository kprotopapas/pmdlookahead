import os
import subprocess
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PMD TS runner', description='Runner for PMD TS', epilog='')
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--horizon', '-t', type=int, default=25)

    parser.add_argument('--n_agents', '-n', type=int, default=10)
    parser.add_argument('--alg', '-a', type=str, default='DPQA')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', '-l', type=float, default=0.1)
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('-d', type=int, default=1)


    args = parser.parse_args()

    size = args.size
    horizon = args.horizon

    alg = args.alg.upper()
    n_agents = args.n_agents
    lr = args.lr
    epochs = args.epochs
    d = args.d

    for i in range(n_agents):
        if f'logistic_{d}' not in os.listdir():
            os.mkdir(f'logistic_{d}')
        
        os.mkdir(f'logistic_{d}/{i}')

        script = \
        f"""#!/bin/bash
#SBATCH -n 1
#SBATCH --time=30:00
#SBATCH --job-name=logistic{d}/{i}
#SBATCH --output=logistic_{d}/{i}/output.out
#SBATCH --error=logistic_{d}/{i}/error.err

module load gcc/8.2.0 python/3.10.4
source cluster-venv/bin/activate
wandb offline
python main.py --alg {alg} -n {n_agents} -s {size} -t {horizon} --epochs {epochs} --wandb --group logistic_v1 -d {d}
"""
        
        with open(f'script_{d}_{i}', 'w') as f:
            f.write(script)
        
        subprocess.run(['sbatch', f'script_{d}_{i}'])

        time.sleep(1)
        os.remove(f'script_{d}_{i}')