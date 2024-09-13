name=naiveblip_sgvqa_lamol

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
        --epochs 1 \
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
        --balance_strategy 'none' \
        --method 'lamol' \
        --use_cap_loss True \
        --show_train_progress True \
        --memory \
        --checkpoint 'snap/naiveblip_sgvqa_lamol/object_BEST'