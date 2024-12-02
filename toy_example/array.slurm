#!/bin/bash
#SBATCH --job-name=vanilla_PINN

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

#SBATCH --time=2:00:00

#SBATCH --array=1-30%4

echo "The reserved GPUs are: $CUDA_VISIBLE_DEVICES"

source env/bin/activate

python3 train.py -c "config/$1/job_${SLURM_ARRAY_TASK_ID}.yaml"

rm config/$1/job_${SLURM_ARRAY_TASK_ID}.yaml
 
