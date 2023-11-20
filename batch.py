import os
import subprocess
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PMD TS runner', description='Runner for PMD TS', epilog='')
    parser.add_argument('file', type=str)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--horizon', '-t', type=int, default=25)

    parser.add_argument('--n_agents', '-n', type=int, default=10)
    parser.add_argument('--alg', '-a', type=str, default='DPQA')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', '-l', type=float, default=0.1)
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('-d', type=int, default=1)

    parser.add_argument('--group', type=str, default='delete')


    args = parser.parse_args()

    file = args.file

    size = args.size
    horizon = args.horizon

    alg = args.alg.upper()
    n_agents = args.n_agents
    lr = args.lr
    epochs = args.epochs
    d = args.d
    
    group = args.group

    for i in range(n_agents):
        if f'{group}_{d}' not in os.listdir('runs'):
            os.mkdir(f'runs/{group}_{d}')
        
        os.mkdir(f'runs/{group}_{d}/{i}')

        script = \
        f"""#!/bin/bash
#SBATCH -n 2
#SBATCH --time=20:00
#SBATCH --job-name={group}/{d}/{i}
#SBATCH --output=runs/{group}_{d}/{i}/output.out
#SBATCH --error=runs/{group}_{d}/{i}/error.err

module load gcc/8.2.0 python/3.10.4
source cluster-venv/bin/activate
wandb offline
python {file}.py --alg {alg} -n 1 -s {size} -t {horizon} --epochs {epochs} --wandb --group {group} -d {d}
"""
        
        with open(f'script_{d}_{i}', 'w') as f:
            f.write(script)
        
        subprocess.run(['sbatch', f'script_{d}_{i}'])

        time.sleep(1)
        os.remove(f'script_{d}_{i}')
