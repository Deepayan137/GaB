#!/bin/bash
#SBATCH --job-name=testBLIP_multi
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (usually, leave at 1)
#SBATCH --cpus-per-task=4       # CPU cores per task
#SBATCH -t 06:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=32G 
#SBATCH --array=0-5
#SBATCH -o logs/test_blip_sgvqa_last_%A_%a.out

# The name of experiment
name='naiveblip_sgvqa_mem_avg_last'

output=snap/$name


PYTHONPATH=$PYTHONPATH:./src \
python src/sgvqa.py \
        --train train \
        --valid val \
        --test val \
        --optim adamw \
        --num_workers 0 \
        --backbone 'Salesforce/blip2-opt-2.7b' \
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 1 \
        --optim 'blip_adamw' \
        --local-rank 0 \
        --eval_blip True \
        --ft_layers 'query_tokens' \
        --checkpoint '' \
        --show_train_progress True \