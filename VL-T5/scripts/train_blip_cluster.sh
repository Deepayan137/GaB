#!/bin/bash
#SBATCH --job-name=trainBLIP1epoch
#SBATCH -A staff
#SBATCH -p gpupart
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (usually, leave at 1)
#SBATCH --cpus-per-task=4       # CPU cores per task
#SBATCH -t 1-00:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=32G 
#SBATCH -o logs/train_blip_1epoch.out
#SBATCH --signal=B:SIGTERM@300

# source /home/deepayan.das/miniconda3/envs/vqacl/etc/profile.d/conda.sh
# conda activate vqacl

name=naiveblip_qtoken

output=snap/$name

# Set environment variables
export PYTHONPATH=$PYTHONPATH:./src

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

#call your program here
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port 64442 \
        src/vqacl.py \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 5 \
        --lr 1e-6 \
        --epochs 1 \
        --num_workers 4 \
        --backbone 'Salesforce/blip2-opt-2.7b' \
        --output "snap/naiveblip_qtoken" \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 1 \
        --from_scratch \
        --optim 'blip_adamw' \
        --m_size 5000 \
        --comp_cate G-1 \
        --now_train \