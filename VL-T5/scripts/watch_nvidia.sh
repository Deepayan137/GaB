#!/bin/bash
#SBATCH -p gpupart   # replace with your partition name
#SBATCH -A staff
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (usually, leave at 1)
#SBATCH --gres=gpu:1 # request for a GPU
#SBATCH -t 00:05:00  # short time just for testing
#SBATCH -o logs/watch.out

nvidia-smi