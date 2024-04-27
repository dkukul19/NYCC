#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR="/path/to/your/data/directory"
# export PYTHONPATH="~/.conda/envs/llava/bin/python"
export GPUS_PER_NODE=1

# MODEL CONFIG
VISION_TOWER=openai/clip-vit-large-patch14
LM_MODEL_NAME=../../../sft_output #LLaVA-RLHF-13b-v1.5-336/sft_model

# DATA CONFIG
PREFERENCE_DATA=../../data/llava_7b_v1_preference.json

# SAVE CONFIG
MODEL_NAME=../../../reward_model

# TRAINING CONFIG
NUM_EPOCHS=1
LEARNING_RATE=2e-5
BATCH_SIZE=4
GRAD_ACCUMULATION=1


python \
    ../../finetune_lora_rm.py \
    --do_train \
    --do_eval \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path $LM_MODEL_NAME \
    --image_folder /datasets/COCO/train2017 \
    --vision_tower $VISION_TOWER \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --model_max_length 2048 \
    --query_len 1280 \
    --response_len 768 \
    --dataset_path $PREFERENCE_DATA \
    --eval_dataset_path $PREFERENCE_DATA \
    --dataset_name "none" \
    --eval_dataset_name "none" \
    --eval_size 500 \
    --bits 16 \
    --lora_r 64 \
    --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --output_dir "$MODEL_NAME" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 10 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --report_to wandb \
    --bf16 True \
    --resume_from_training True \
    --reward_prompt_file "../../prompts/fact_rlhf_reward_prompt.txt" \
    --image_to_caption_file "../../data/image_to_caption.json" \
    --image_aspect_ratio 'pad'
