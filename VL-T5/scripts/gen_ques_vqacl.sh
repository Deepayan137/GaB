#!/bin/bash
#SBATCH -J EVAL
#SBATCH -p long-disi
#SBATCH --gres gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH --array=1-9
#SBATCH -o logs/bal_qtype_%a.out  # %A for job ID, %a for array task ID
# # Load Python environment or any other dependencies
# module load python/3.8  # Example module, adjust as per your setup

# # The name of experiment
#python -m src.analysis.vqacl_gen_ques
#python -m src.analysis.vqacl_question_distribution
python -m src.analysis.vqacl_create_balanced_rehearsal
# python -m src.analysis.vqacl_question_distribution2
#python -m src.analysis.vqacl_gen_ques_past_images
