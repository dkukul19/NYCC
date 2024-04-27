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
set -e
set -x

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
export DATA_DIR="/path/to/your/data/directory"
export GPUS_PER_NODE=1

# MODEL CONFIG
POLICY_BASE_MODEL_NAME=../../../sft_output_2
RM_BASE_MODEL_NAME=../../../sft_output_2

POLICY_LORA=../../../init_model
RM_LORA=../../../reward_model/checkpoint-2200  # we use early stopping

# SAVE CONFIG
MODEL_NAME=../../../rlhf_model

# TRAINING CONFIG
LEARNING_RATE=1e-5
KL_COEF=0.1
EPOCH=1
ROLLOUT_BATCH_SIZE=128
STEP_BATCH_SZIE=64
ROLLOUT_PER_DEVICE_BATCH_SIZE=16
REWARD_MODEL_PER_DEVICE_BATCH_SIZE=8
STEP_PER_DEVICE_BATCH_SIZE=8
NOPTEPOCHS=2

# FACT-RLHF CONFIG
INCOMPLETE_RESPONSE=-8.0
LENGTH_BONUS=-10.0
CORRECT_BONUS=2.0

python ../../finetune_lora_ppo.py \
    --do_train \
    --seed 42 \
    --step_batch_size $STEP_BATCH_SZIE \
    --step_per_device_batch_size $STEP_PER_DEVICE_BATCH_SIZE \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --rollout_per_device_batch_size $ROLLOUT_PER_DEVICE_BATCH_SIZE \
    --reward_model_per_device_batch_size $REWARD_MODEL_PER_DEVICE_BATCH_SIZE \
    --base_model_name "$POLICY_BASE_MODEL_NAME" \
    --reward_base_model_name "$RM_BASE_MODEL_NAME" \
    --policy_model_name_or_path "$POLICY_LORA" \
    --reward_model_name_or_path "$RM_LORA" \
    --learning_rate $LEARNING_RATE \
    --init_value_with_reward True \
    --warmup_steps 5 \
    --dataset_path ../../data/llava_ppo50k-aokvqa12k-vqa10k.json \
    --train_splits "train" \
    --output_dir "$MODEL_NAME" \
    --total_epochs $EPOCH \
    --group_by_length False \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 100000 \
    --weight_decay 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --bf16 True \
    --penalty_reward_value $INCOMPLETE_RESPONSE \
    --length_bonus_score $LENGTH_BONUS \
    --correct_bonus_score $CORRECT_BONUS \
    --relative_stop_token_penalty True \
    --penalize_no_stop_token True \
    --resume_from_training True \
    --kl_coef $KL_COEF \
    --max_grad_norm 1.0 \
    --whitening_async_stats "full_batch" \
    --clean_tokens_after_eos True \
    --temperature 1.0 \
    --whiten_rewards False \
    --model_max_length 2048 \
    --query_len 128 \
    --response_len 896 \
    --noptepochs $NOPTEPOCHS \
    --image_folder /datasets/COCO/train2017 \
    --vision_tower different \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --reward_prompt_file "../../prompts/fact_rlhf_reward_prompt.txt" \
    --image_to_caption_file "../../data/image_to_caption.json" \
    --image_aspect_ratio 'pad'
