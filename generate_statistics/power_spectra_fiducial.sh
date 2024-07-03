#!/bin/bash
#SBATCH --job-name=power_fiducial
#SBATCH --nodes=1
#SBATCH --account=rrg-wperciva
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=35G
#SBATCH --array=0-249

n_per_submit=60
start_idx=$((SLURM_ARRAY_TASK_ID * n_per_submit))
echo $start_idx

python densitysplit_power_fiducial.py --start_idx $start_idx --n $n_per_submit
