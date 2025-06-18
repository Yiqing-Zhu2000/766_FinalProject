#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-03:00
#SBATCH --output=test4-%j.out

# useful environment check for ComputeCanada

module load python/3.10  # Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch --no-index
pip install transformers --no-index
pip install "accelerate>=0.26.0" --no-index    # test to temp create env
pip install "transformers[torch]" --no-index

python /home/yzhu439/pytorchGPUTest1/766_FinalProject/train_bert.py