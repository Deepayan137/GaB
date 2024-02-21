#!/bin/bash
#SBATCH --job-name=testBLIP_comp 
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (usually, leave at 1)
#SBATCH --cpus-per-task=1       # CPU cores per task
#SBATCH -t 07:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=32G 
#SBATCH -o logs/test_blip_output.out

name=checkpoint

output=snap/$name

# Set environment variables
export PYTHONPATH=$PYTHONPATH:./src
#call your program here
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port 64444 \
        src/vqacl.py \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 1e-4 \
        --epochs 3 \
        --num_workers 4 \
        --backbone 'Salesforce/blip2-opt-2.7b' \
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 1 \
        --valid_batch_size 1 \
        --optim 'blip_adamw' \
        --eval_blip True

        