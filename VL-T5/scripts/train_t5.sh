# The name of experiment
name=vqaclt5

output=snap/$name


PYTHONPATH=$PYTHONPATH:./src \
python src/vqacl.py \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 5 \
        --lr 1e-4 \
        --epochs 1 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 80 \
        --from_scratch \
        --optim 'adamw' \
        --m_size 5000 \
        --comp_cate G-1 \
        --now_train \
        --local-rank 0 \
        --show_train_progress True \
        --use_class_hierarchy True \
        --train_from_scratch False \
        --ft_layers 'query_tokens' \
        --memory \
        # --checkpoint "snap/vqaclt5/q_recognition_LAST"