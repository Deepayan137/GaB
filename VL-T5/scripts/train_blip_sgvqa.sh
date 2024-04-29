# The name of experiment
name=naiveblip_sgvqa_indivigual

output=snap/$name


PYTHONPATH=$PYTHONPATH:./src \
python src/sgvqa.py \
--train train \
--valid val \
--test val \
--optim adamw \
--warmup_ratio 0.05 \
--clip_grad_norm 5 \
--lr 1e-6 \
--epochs 3 \
--num_workers 4 \
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
--show_train_progress True \
--train_from_scratch False \
--ft_layers 'query_tokens' \
--blip_model "naiveblip" \
--scenario "function" \
--memory \
--checkpoint "snap/naiveblip_sgvqa_mem/logical_BEST.pth"