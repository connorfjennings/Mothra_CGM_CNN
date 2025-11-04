#!/bin/bash
#SBATCH --job-name=cutout
#SBATCH -N 1                   # single node
#SBATCH -n 16                  # == number of MPI tasks (ranks)
#SBATCH --cpus-per-task=1      # 1 core per rank
#SBATCH --mem=256G             # adjust as needed /bigmem if huge halos
#SBATCH --output=cutoutshard_out.txt
#SBATCH --partition=day
#SBATCH --time=1:00:00

module reset
module restore TNG_h5py

# In your sbatch script before mpirun:
export OMP_NUM_THREADS=1

# Recommended for parallel HDF5 on many clusters:
export HDF5_COLL_METADATA_WRITE=1
export HDF5_MPI_OPT_TYPES_ENV=1

# If your site recommends it for GPFS/Isilon (often helps):
export HDF5_USE_FILE_LOCKING=FALSE


mpirun -n 16 python make_gas_sphere_cutout_shards.py
