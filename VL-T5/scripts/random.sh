#!/bin/bash
#SBATCH --job-name=gen_captions_com-causal
#SBATCH -p long-disi
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (usually, leave at 1)
#SBATCH --cpus-per-task=1       # CPU cores per task
#SBATCH -t 12:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=20G 
#SBATCH -o logs/cap.out


python -m src.analysis.subset