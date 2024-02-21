# The name of experiment
name=blip2

output=snap/$name


PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 6666 \
    src/vqacl.py \
        --multiGPU \
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
        --batch_size 4 \
        --valid_batch_size [8 \
        --from_scratch \
        --memory \
        --optim 'blip_adamw' \
        --m_size 5000 \
        --comp_cate G-1 \
        --now_train
