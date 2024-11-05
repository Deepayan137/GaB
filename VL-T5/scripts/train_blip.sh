#!/bin/bash
if [ -z "$1" ]; then
    name='naiveblip_cl_balanced_cluster_7_5k' # default value if no argument is provided
else
    name="$1"
fi

if [ -z "$2" ]; then
    balance_strategy='cluster' # default value if no argument is provided
else
    balance_strategy="$2"
fi

output=snap/$name


# Set environment variables
export PYTHONPATH=$PYTHONPATH:./src

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
        --batch_size 80 \
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
        --balance_strategy "${balance_strategy}" \
        --use_cap_loss False \
        --memory \
        --checkpoint "${output}/q_recognition_LAST"