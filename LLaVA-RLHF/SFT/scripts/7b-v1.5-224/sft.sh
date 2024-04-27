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
DATA_PATH=../../../data/new_data.json

LOCAL_RANK=0,1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --use-env \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node-rank=0 \
    ../../train/train.py \
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
    --output_dir ../../../output_directory \
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
    --model_max_length 50 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --image_aspect_ratio 'pad'
