# The name of experiment
name=naiveblip_qtoken

output=snap/$name


PYTHONPATH=$PYTHONPATH:./src \
python src/vqacl.py \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 5 \
        --lr 1e-5 \
        --epochs 1 \
        --num_workers 4 \
        --backbone 'Salesforce/blip2-opt-2.7b' \
        --output 'snap/naiveblip_qtoken' \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 1 \
        --optim 'blip_adamw' \
        --m_size 5000 \
        --comp_cate G-1 \
        --now_train \
        --local-rank 0 \
        # --checkpoint 'snap/naiveblip_cl/q_recognition_LAST.pth' \
        # --load True \
        # --memory
