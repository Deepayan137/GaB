#!/bin/bash
#SBATCH --job-name=train_naiveblip_sgvqa_rebut
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (usually, leave at 1)
#SBATCH --cpus-per-task=4       # CPU cores per task
#SBATCH -t 1-00:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=64G 
#SBATCH -o logs/train_naiveblip_sgvqa_cluster_run1_final.out


name=naiveblip_sgvqa_cluster_run1

output=snap/$name

# Set environment variables
export PYTHONPATH=$PYTHONPATH:./src

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

#call your program here
python -m src.sgvqa \
        --train train \
        --valid val \
        --test val \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 5 \
        --lr 1e-4 \
        --epochs 10 \
        --num_workers 4 \
        --backbone 'Salesforce/blip2-opt-2.7b' \
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 32 \
        --valid_batch_size 1 \
        --from_scratch \
        --optim 'blip_adamw' \
        --m_size 5000 \
        --comp_cate G-1 \
        --now_train \
        --local-rank 0 \
        --ft_layers 'query_tokens' \
        --blip_model "naiveblip" \
        --scenario "function" \
        --sequence "oarlks" \
        --train_from_scratch False \
        --use_gen_data True \
        --balance_strategy 'cluster' \
        --method 'no_ents' \
        --use_cap_loss False \
        --memory \
        --checkpoint 'snap/naiveblip_sgvqa_cluster_run1/logical_BEST' \