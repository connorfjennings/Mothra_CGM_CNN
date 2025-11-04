#!/bin/bash
#SBATCH --job-name=unet_train
#SBATCH --output=unet_train_out.txt
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

srun python allgas_vel_train.py

