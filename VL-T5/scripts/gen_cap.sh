#!/bin/bash
#SBATCH -J EVAL
#SBATCH -p boost_usr_prod
#SBATCH --gres gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --array=0-1
#SBATCH -o logs/gen_cap_%a.out  # %A for job ID, %a for array task ID
# # Load Python environment or any other dependencies
# module load python/3.8  # Example module, adjust as per your setup

# # The name of experiment
python -m src.analysis.gen_cap_sgvqa
# python -m src.analysis.make_feat