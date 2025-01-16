#!/bin/bash

#SBATCH --mem-per-cpu=1G

#SBATCH --time=96:00:00

echo "The reserved GPUs are: $CUDA_VISIBLE_DEVICES"

source ~/home/mbabic/PINNs_fixed_points/env/bin/activate

python3 run_experiments.py