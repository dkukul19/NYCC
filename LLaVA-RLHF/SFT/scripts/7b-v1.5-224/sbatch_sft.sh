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
#SBATCH --output=slurm_logs/test-%J.log

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

#export NCCP_P2P_DISABLE=1

#export CUDA_HOME=/usr/local/cuda
#export PATH=$CUDA_HOME/bin:$PATH
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION=vicuna-v1-3-7b
################## VICUNA ##################

LM_MODEL_CKPT=lmsys/vicuna-7b-v1.3
###/kuacc/users/dkukul19/hpc_run/LLaVA-RLHF/LLaVa-RLHF_model_checkpoints   #lmsys/vicuna-7b-v1.5
MM_CKPT=../../../MM_projector/llava-pretrain-vicuna-7b-v1.3/mm_projector.bin  #/shared/llava-$MODEL_VERSION-pretrain/mm_projector.bin
DATA_PATH=../../../data/new_data_combined.json

deepspeed ../../train/train.py \
    --deepspeed ../zero3.json \
    --model_name_or_path $LM_MODEL_CKPT \
    --version $PROMPT_VERSION \
    --data_path ${DATA_PATH} \
    --image_folder /datasets/COCO/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter $MM_CKPT \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ../../../sft_output_2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1280 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --save_full_model True \
    --report_to wandb \
    --image_aspect_ratio 'pad' \
    --lora_enable True
