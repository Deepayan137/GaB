# The name of experiment
name=naiveblip_cl_test

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
        --num_workers 0 \
        --backbone 'Salesforce/blip2-opt-2.7b' \
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 64 \
        --valid_batch_size 1 \
        --from_scratch \
        --optim 'blip_adamw' \
        --m_size 5000 \
        --comp_cate G-1 \
        --now_train \
        --local-rank 0 \
        --show_train_progress True \
        --train_from_scratch True \
        --ft_layers 'query_tokens' \
        --blip_model "naiveblip" \
        --use_class_hierarchy True \
        --use_gen_data True \
        --use_cap_loss True \
        --method 'lamol' \
        --memory \
        --checkpoint 'snap/naiveblip_cl_test/q_recognition_LAST.pth'