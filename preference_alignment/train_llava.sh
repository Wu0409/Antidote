# WE USE SWIFT FRAMEWORK (https://github.com/modelscope/ms-swift) to post-train LVLMs

CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift rlhf \
    --model_type llava1_5-7b-instruct \
    --dtype bf16 \
    --num_train_epochs 1 \
    --learning_rate 2e-6 \
    --warmup_ratio 0.0 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --sft_type lora \
    --lora_rank 128 \
    --lora_alpha 256 \
    --beta 0.1 \
    --rpo_alpha 0.00 \
    --freeze_vit true \
    --max_length 1024 \
    --deepspeed default-zero2 \
    --dataset XXX/inference/demo/train/dpo_cpq_list.jsonl \
              XXX/inference/demo/train/dpo_tpq_list.jsonl \
              XXX/inference/demo/train/dpo_desc_list.jsonl \
              XXX/inference/demo/train/dpo_pope_list.jsonl \
    --output_dir output/llava1_5-7b-instruct/antidote_lora \
    --add_output_dir_suffix False \
    --save_total_limit 1 \
    --seed 0 \
    --eval_steps 5000
