import os
import subprocess


if __name__ == '__main__':

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
        python main.py --alg DPQA -n 10 -s 8 -t 10 --epochs 50 --wandb --group logistic_v1 -d {d}
        """
        
        with open('script', 'w') as f:
            f.write(script)
        
        subprocess.run(['sbatch', 'script'])