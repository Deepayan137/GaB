#!/bin/bash
#SBATCH --job-name=train_naiveblip_sgvqa_mem
#SBATCH -p long-disi
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (usually, leave at 1)
#SBATCH --cpus-per-task=4       # CPU cores per task
#SBATCH -t 2-00:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=32G 
#SBATCH -o logs/train_naiveblip_sgvqa_mem.out
#SBATCH --signal=B:SIGTERM@300


name=naiveblip_sgvqa_mem

output=snap/$name

# Set environment variables
export PYTHONPATH=$PYTHONPATH:./src

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

#call your program here
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port 62225 \
        src/sgvqa.py \
        --train train \
        --valid val \
        --test val \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 15 \
        --num_workers 0 \
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
        --train_from_scratch False \
        --ft_layers 'query_tokens' \
        --blip_model "naiveblip" \
        --scenario "function" \
        --memory \
        --checkpoint 'snap/naiveblip_sgvqa_mem/logical_BEST.pth'
