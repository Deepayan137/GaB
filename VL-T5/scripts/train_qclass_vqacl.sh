#!/bin/bash
#SBATCH -J train_qclassifier
#SBATCH -p long-disi
#SBATCH --gres gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --array=1-9
#SBATCH -o logs/vqacl_balanced%a.out

# python -m src.analysis.vqacl_question_distribution
python -m src.analysis.vqacl_create_balanced_rehearsal
# python -m src.analysis.vqacl_question_distribution