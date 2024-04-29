#!/bin/bash
#SBATCH -J EVAL
#SBATCH -p boost_usr_prod
#SBATCH --gres gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o logs/test_sgvqa.out
#SBATCH -e logs/test_sgvqa.err

# The name of experiment
name='naiveblip_sgvqa_mem'

output=snap/$name


PYTHONPATH=$PYTHONPATH:./src \
python src/sgvqa.py \
        --train train \
        --valid val \
        --test val \
        --optim adamw \
        --num_workers 4 \
        --backbone 'Salesforce/blip2-opt-2.7b' \
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 1 \
        --optim 'blip_adamw' \
        --local-rank 0 \
        --eval_blip True \
        --ft_layers 'query_tokens' \
        --checkpoint 'snap/naiveblip_sgvqa_mem/scenetext_BEST' \
        --show_train_progress True