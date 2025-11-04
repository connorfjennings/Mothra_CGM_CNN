#!/bin/bash
#SBATCH --job-name=coldgas_vel_train
#SBATCH --output=coldgas_vel_out.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=24:00:00   
#SBATCH --mem=32G
#SBATCH --gpus=a5000:1

module purge
module load miniconda
conda activate torch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python seg_models_train.py packed_aug8_coldvel coldgas_vel

