# The name of experiment
name='vqaclblip'

output=snap/$name


PYTHONPATH=$PYTHONPATH:./src \
python src/vqacl.py \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 1e-4 \
        --epochs 3 \
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
        # --checkpoint 'snap/vqaclblip/q_location_LAST' \