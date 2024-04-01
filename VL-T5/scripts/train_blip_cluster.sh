#!/bin/bash
#SBATCH --job-name=train_naiveblip_multi2
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (usually, leave at 1)
#SBATCH --cpus-per-task=4       # CPU cores per task
#SBATCH -t 1-00:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=32G 
#SBATCH -o logs/train_naiveblip_multi2.out
#SBATCH --signal=B:SIGTERM@300


name=naiveblip_multi

output=snap/$name

# Set environment variables
export PYTHONPATH=$PYTHONPATH:./src

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

#call your program here
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port 62226 \
        src/vqacl.py \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 5 \
        --lr 1e-6 \
        --epochs 3 \
        --num_workers 4 \
        --backbone 'Salesforce/blip2-opt-2.7b' \
        --output $output ${@:2}  \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 1 \
        --from_scratch \
        --optim 'blip_adamw' \
        --m_size 5000 \
        --comp_cate G-1 \
        --now_train \
        --train_from_scratch False \
        --show_train_progress False \
        --use_class_hierarchy False \
        --ft_layers 'query_tokens' \
        --train_multi True \
        --blip_model "naiveblip" \
        --checkpoint 'snap/naiveblip_multi/q_causal1'
