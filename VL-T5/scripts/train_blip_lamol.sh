#!/bin/bash
#SBATCH --job-name=train_naiveblip_lamol
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (usually, leave at 1)
#SBATCH --cpus-per-task=4       # CPU cores per task
#SBATCH -t 1-00:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=64G 
#SBATCH -o logs/train_naiveblip_lamol3.out

name=naiveblip_cl_lamol

output=snap/$name

# Set environment variables
export PYTHONPATH=$PYTHONPATH:./src

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

#call your program here
python src/vqacl.py \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 5 \
        --lr 1e-4 \
        --epochs 3 \
        --num_workers 4 \
        --backbone 'Salesforce/blip2-opt-2.7b' \
        --output $output ${@:2}  \
        --num_beams 5 \
        --batch_size 32 \
        --valid_batch_size 1 \
        --from_scratch \
        --optim 'blip_adamw' \
        --m_size 5000 \
        --comp_cate G-1 \
        --now_train \
        --local-rank 0 \
        --show_train_progress False \
        --train_from_scratch False \
        --ft_layers 'query_tokens' \
        --blip_model "naiveblip" \
        --use_gen_data True \
        --balance_strategy 'none' \
        --use_cap_loss True \
        --method 'lamol' \
        --memory \
        --checkpoint 'snap/naiveblip_cl_lamol/q_action_LAST.pth'