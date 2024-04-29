#!/bin/bash
#SBATCH -J EVAL
#SBATCH -p boost_usr_prod
#SBATCH --gres gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/train_naiveblip_sgvqa_indi.out


# source ~/.bashrc
# eval "$(conda shell.bash hook)"
# conda activate ul

# module load cuda

name=naiveblip_sgvqa_indivigual

output=snap/$name

# Set environment variables
export PYTHONPATH=$PYTHONPATH:./src

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

#call your program here
python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port 62222 \
src/sgvqa.py \
--train train \
--valid val \
--test val \
--optim adamw \
--warmup_ratio 0.05 \
--clip_grad_norm 5 \
--lr 5e-5 \
--epochs 3 \
--num_workers 0 \
--backbone 'Salesforce/blip2-opt-2.7b' \
--output $output ${@:2} \
--num_beams 5 \
--batch_size 1 \
--valid_batch_size 1 \
--from_scratch \
--optim 'blip_adamw' \
--m_size 5000 \
--comp_cate G-1 \
--now_train \
--local-rank 0 \
--train_from_scratch False \
--ft_layers 'query_tokens' \
--blip_model "naiveblip" \
--scenario "function"