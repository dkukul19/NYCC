#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

export GPUS_PER_NODE=1

# MODEL CONFIG
VISION_TOWER=openai/clip-vit-large-patch14
LM_MODEL_NAME=../../../sft_output_2 #LLaVA-RLHF-7b-v1.5-224/sft_model

# SAVE CONFIG
MODEL_NAME=../../../init_model

# TRAINING CONFIG
NUM_EPOCHS=1
LEARNING_RATE=1e-4
BATCH_SIZE=8
GRAD_ACCUMULATION=2

deepspeed ../../finetune_lora_sft_ds.py \
    --deepspeed scripts/zero2.json \
    --do_train \
    --do_eval \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path $LM_MODEL_NAME \
    --image_folder /datasets/COCO/train2017 \
    --vision_tower $VISION_TOWER \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --query_len 1280 \
    --response_len 768 \
    --dataset ../../data/llava_reward10k-aokvqa5k.json \
    --dataset_format "v1" \
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
    --save_steps 1000000 \
    --save_total_limit 1 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --report_to "wandb" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --image_aspect_ratio 'pad'
