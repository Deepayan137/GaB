#!/bin/bash
#SBATCH -J EVAL
#SBATCH -p boost_usr_prod
#SBATCH --gres gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --array=1-4
#SBATCH -o logs/balanced_rehearsal_%a.out  # %A for job ID, %a for array task ID
# # Load Python environment or any other dependencies
# module load python/3.8  # Example module, adjust as per your setup

# # The name of experiment
# python -m src.analysis.question_distribution
python -m src.analysis.create_balanced_rehearsal
# python -m src.analysis.question_distribution_ver2
# python -m src.analysis.gen_ques
# python -m src.analysis.gen_ques