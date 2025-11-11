#!/bin/bash
#SBATCH --job-name=coldgas_vel_train
#SBATCH --output=coldgas_vel_out.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=3:00:00   
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --constraint="a5000|v100|a100"

module purge
module load miniconda
module load CUDA
conda activate torch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python seg_models_train.py packed_aug8_coldvel coldgas_vel

