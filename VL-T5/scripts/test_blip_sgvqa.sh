# The name of experiment
name='naiveblip_sgvqa_mem_new'

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
        --batch_size 32 \
        --valid_batch_size 1 \
        --optim 'blip_adamw' \
        --local-rank 0 \
        --eval_blip True \
        --ft_layers 'query_tokens' \
        --checkpoint 'snap/naiveblip_sgvqa_mem_new/object_BEST' \
        --show_train_progress True