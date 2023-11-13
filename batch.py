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

    ds = [1]
    ds.extend(range(5, 25, 5))
    for d in ds:
        os.mkdir(f'logistic{d}')
        script = f"""
        #!/bin/bash

        #SBATCH -n 4
        #SBATCH --time=1:00:00
        #SBATCH --job-name=logistic{d}/logistic{d}
        #SBATCH --output=logistic{d}/logistic{d}.out
        #SBATCH --error=logistic{d}/logistic{d}.err

        module load gcc/8.2.0 python/3.10.4
        source cluster-venv/bin/activate
        python main.py --alg {alg} -n {n_agents} -s {size} -t {horizon} --epochs {epochs} --wandb --group logistic_v1 -d {d}
        """
        
        with open(f'script{d}', 'w') as f:
            f.write(script)
        
        subprocess.run(['sbatch', 'script'])

        time.sleep(1)
        os.remove(f'script{d}')