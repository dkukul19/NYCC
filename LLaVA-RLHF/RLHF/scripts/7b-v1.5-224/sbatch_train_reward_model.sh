#!/bin/bash
# 
#CompecTA (c) 2018
# 
# You should only work under the /scratch/users/<username> directory.
#
# Jupyter job submission script
#
# TODO:
#   - Set name of the job below changing "JupiterNotebook" value.
#   - Set the requested number of nodes (servers) with --nodes parameter.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter. (Total accross all nodes)
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - mid   : For jobs that have maximum run time of 1 day..
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input/output file names below.
#   - If you do not want mail please remove the line that has --mail-type and --mail-user. If you do want to get notification emails, set your email address.
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch jupyter_submit.sh
#
# -= Resources =-
#

#SBATCH --job-name=JupiterNotebook
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --mem=100G
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=1-0:0:0
#SBATCH --output=slurm_logs/reward_model-%J.log

# Please read before you run: http://login.kuacc.ku.edu.tr/#h.3qapvarv2g49

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

# Load Anaconda
echo "======================="
echo "Loading Anaconda Module..."
#module load anaconda/2.7
module load gcc/11.2.0
module load cuda/11.8.0
module load cudnn/8.2.0/cuda-11.X
echo "======================="

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

# MODEL CONFIG
VISION_TOWER=openai/clip-vit-large-patch14
LM_MODEL_NAME=../../../sft_output_2  #/kuacc/users/oince22/.cache/huggingface/hub/models--zhiqings--LLaVA-RLHF-7b-v1.5-224/snapshots/e209b5158897700108cefae829393079dbea0416/sft_model #../../../sft_output_2

# DATA CONFIG
PREFERENCE_DATA=../../data/llava_7b_v1_preference.json

# SAVE CONFIG
MODEL_NAME=../../../reward_model

# TRAINING CONFIG
NUM_EPOCHS=1
LEARNING_RATE=2e-5
BATCH_SIZE=4
GRAD_ACCUMULATION=1

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
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
    --logging_steps 5 \
    --report_to "wandb" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --reward_prompt_file "../../prompts/fact_rlhf_reward_prompt.txt" \
    --image_to_caption_file "../../data/image_to_caption.json" \
    --image_aspect_ratio 'pad'