#!/bin/bash
#SBATCH --job-name=testBLIP 
#SBATCH -p long-disi
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (usually, leave at 1)
#SBATCH --cpus-per-task=1       # CPU cores per task
#SBATCH -t 1-00:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=20G 
#SBATCH -o logs/test_blip_cluster_loop7.out

# List of experiment names
# names=('naiveblip_qtoken' 'naiveblip_cl_qtoken' 'naiveblip_nohier_qtoken' 'naiveblip_cl_nohier_qtoken')
#names=( 'naiveblip_full' 'naiveblip_nohier_full' 'naiveblip_cl_full' 'naiveblip_cl_nohier_full')
# names=('naiveblip_scratch_nohier_qtoken' 'naiveblip_scratch_qtoken')
#names=('naiveblip_qtoken_1ep' 'naiveblip_cl_qtoken_1ep')
# names=('naiveblip_scratch_nohier_qtoken_1e' 'naiveblip_scratch_qtoken_1e')
names=('naiveblip_ewc')
# List of all tasks

All_task=('q_recognition' 'q_location' 'q_judge' 'q_commonsense' 'q_count' 'q_action' 'q_color' 'q_type' 'q_subcategory')

# Loop over each name
for name in "${names[@]}"; do
    # Determine ft_layers based on name suffix
    if [[ $name == *"full" ]]; then
        ft_layers='full'
    else
        ft_layers='query_tokens'
    fi

    # Loop over each task
    for task in "${All_task[@]}"; do
        output="snap/$name"
        checkpoint="snap/$name/${task}_LAST"

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
            --output $output \
            --num_beams 5 \
            --batch_size 80 \
            --valid_batch_size 1 \
            --optim 'blip_adamw' \
            --local-rank 0 \
            --eval_blip True \
            --checkpoint $checkpoint \
            --ft_layers 'query_tokens'\
            ${@:2}
    done
done
