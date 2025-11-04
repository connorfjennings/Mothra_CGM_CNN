#!/bin/bash
#SBATCH --job-name=vmap
#SBATCH --time=1:00:00 --ntasks 1 --mem=128G
#SBATCH --output=out.txt

module load miniconda
conda activate TNG50
python velocity_map.py