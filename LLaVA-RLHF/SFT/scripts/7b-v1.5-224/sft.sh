

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

module load gcc/11.2.0
module load cuda/11.8.0
module load cudnn/8.2.0/cuda-11.X

#export NCCP_P2P_DISABLE=1


nvidia-smi
python --version
nvcc --version
gcc --version
# Uncomment and set the following variables correspondingly to run this script:
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/kuacc/apps/cudnn/v8.0.4_CUDA_10.2/lib64

#export CUDA_HOME=/usr/local/cuda
#export PATH=$CUDA_HOME/bin:$PATH
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


export HF_HOME=/shared/sheng/huggingface
export TRANSFORMERS_CACHE=/kuacc/users/dkukul19/hpc_run/LLaVA-RLHF/cache

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION=vicuna-v1-5-7b
################## VICUNA ##################
# change model version to 1.3
################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

LM_MODEL_CKPT=/kuacc/users/dkukul19/hpc_run/LLaVA-RLHF/LLaVA-RLHF/LLaVA-RLHF_checkpoints_7b-1.3/vicuna-7b-v1.3
###/kuacc/users/dkukul19/hpc_run/LLaVA-RLHF/LLaVa-RLHF_model_checkpoints   #lmsys/vicuna-7b-v1.5
MM_CKPT=/kuacc/users/dkukul19/hpc_run/LLaVA-RLHF/MM_projector/mm_projector.bin  #/shared/llava-$MODEL_VERSION-pretrain/mm_projector.bin
DATA_PATH=/kuacc/users/dkukul19/hpc_run/LLaVA-RLHF/Data_json/mix-llava-sft90k-vqav2_83k-okvqa_16k-flickr_23k.json

deepspeed /kuacc/users/dkukul19/hpc_run/LLaVA-RLHF/LLaVA-RLHF/SFT/train/train.py \
    --deepspeed "../zero3.json" \
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
    --output_dir /kuacc/users/dkukul19/hpc_run/LLaVA-RLHF/LLaVA-RLHF/output_directory \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 100 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --image_aspect_ratio 'pad'
