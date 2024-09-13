#!/bin/bash
#SBATCH --job-name=testBLIP_multi
#SBATCH -p long-disi
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (usually, leave at 1)
#SBATCH --cpus-per-task=4       # CPU cores per task
#SBATCH -t 06:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=32G 
#SBATCH -o logs/test_blip_balanced_cluster_7_5k_final%a.out
#SBATCH --array=9
name='naiveblip_balanced_cluster_7_5k_run_final'

output=snap/$name

# Set environment variables
export PYTHONPATH=$PYTHONPATH:./src
#call your program here
python src/vqacl.py \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --num_workers 4 \
        --backbone 'Salesforce/blip2-opt-2.7b' \
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 1 \
        --optim 'blip_adamw' \
        --eval_blip True \
        --ft_layers 'query_tokens' \
        --checkpoint '' \
        --local-rank 0 \
        --show_train_progress False
        
