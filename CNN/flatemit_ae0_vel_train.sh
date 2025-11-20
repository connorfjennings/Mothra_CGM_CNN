#!/bin/bash
#SBATCH --job-name=flatgas_ae0_vel_train
#SBATCH --output=flatemit_ae0_vel_out.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=2:00:00   
#SBATCH --mem=30G
#SBATCH --gpus=1
#SBATCH --constraint="a5000|v100|a100"

module purge
module load miniconda
module load CUDA
conda activate torch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python seg_models_train.py packed_aug8_flatemitvel flatemit_vel

