#!/bin/bash
#SBATCH --job-name=multihead_C8
#SBATCH --output=multihead_C8_out.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=2:00:00   
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint="a5000|v100|a100"

module purge
module load miniconda
module load CUDA
conda activate torch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python seg_models_multihead_train.py coldvel_1e-22mask_C8_20_200_profile.pkl coldgas_multihead_C8

