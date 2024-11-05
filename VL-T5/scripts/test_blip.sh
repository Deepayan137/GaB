#!/bin/bash
# Check if a name was passed as an argument, if not, use a default value
if [ -z "$1" ]; then
    name='naiveblip_cl_balanced_cluster_7_5k' # default value if no argument is provided
else
    name="$1"
fi
output=snap/$name
# Set environment variables
export PYTHONPATH=$PYTHONPATH:./src
#call your program here
python src/vqacl.py \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --num_workers 4 \
        --backbone 'Salesforce/blip2-opt-2.7b' \
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 1 \
        --optim 'blip_adamw' \
        --eval_blip True \
        --ft_layers 'query_tokens' \
        --checkpoint '' \
        --local-rank 0 \
        --show_train_progress False