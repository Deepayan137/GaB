#!/bin/bash
#SBATCH --job-name=testBLIP_sgvqa 
#SBATCH -p long-disi
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (usually, leave at 1)
#SBATCH --cpus-per-task=1       # CPU cores per task
#SBATCH -t 06:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=20G 
#SBATCH -o logs/test_blip_cluster_loop_sgvqa.out

name='naiveblip_sgvqa'
# List of all tasks

All_task=("object" "attribute" "relation" "logical" "knowledge" "scenetext")

# # Loop over each name
# for name in "${names[@]}"; do
#     # Determine ft_layers based on name suffix
#     if [[ $name == *"full" ]]; then
#         ft_layers='full'
#     else
#         ft_layers='query_tokens'
#     fi

# Loop over each task
for task in "${All_task[@]}"; do
    # Skip the iteration if task is 'object'
    # if [ "$task" = "object" ]; then
    #     continue
    # fi
    output="snap/$name"
    checkpoint="snap/$name/${task}_BEST"

    PYTHONPATH=$PYTHONPATH:./src \
    python src/sgvqa.py \
        --train train \
        --valid val \
        --test val \
        --optim adamw \
        --num_workers 4 \
        --backbone 'Salesforce/blip2-opt-2.7b' \
        --output $output ${@:2} \
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

