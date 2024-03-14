# The name of experiment
name=naiveblip_test

output=snap/$name


PYTHONPATH=$PYTHONPATH:./src \
python src/vqacl.py \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 5 \
        --lr 1e-6 \
        --epochs 2 \
        --num_workers 4 \
        --backbone 'Salesforce/blip2-opt-2.7b' \
        --output "snap/naiveblip_test" \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 1 \
        --from_scratch \
        --optim 'blip_adamw' \
        --m_size 5000 \
        --comp_cate G-1 \
        --now_train \
        --local-rank 0 \
        --show_train_progress True \
        --log_all_runs True \
        --use_class_hierarchy False \
        --memory
        # --checkpoint "snap/naiveblip_layers/q_location_LAST.pth"