#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=2G

#SBATCH --time=72:00:00

echo "The reserved GPUs are: $CUDA_VISIBLE_DEVICES"

source ~/home/mbabic/PINNs_fixed_points/env/bin/activate

python3 run_experiments.py