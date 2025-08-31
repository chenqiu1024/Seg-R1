#!/bin/bash


# The latest vllm==0.7.3 is required for this script: pip3 install vllm==0.7.3
# The latest transformers is required too, install by: pip install git+https://github.com/huggingface/transformers.git@a40f1ac602fe900281722254c52ce3773f28eb0e



export PATH="/root/autodl-tmp/envs/seg-r1/bin:$PATH"
export DEBUG_MODE="true"
export LOG_PATH="./vllm_run.txt"
export WANDB_DISABLED="true"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/root/autodl-tmp/works/Seg-R1/.hf_home"
export HF_DATASETS_CACHE="/root/autodl-tmp/works/Seg-R1/.hf_cache"
export TMPDIR="/root/autodl-tmp/works/Seg-R1/.tmp"
export XDG_CACHE_HOME="/root/autodl-tmp/works/Seg-R1/.cache"
export TRITON_CACHE_DIR="/root/autodl-tmp/works/Seg-R1/.triton"
export TORCHINDUCTOR_CACHE_DIR="/root/autodl-tmp/works/Seg-R1/.torchinductor"

QWEN_PATH="/root/autodl-tmp/works/Seg-Zero/pretrained_models/Qwen2.5-VL-7B-Instruct"
HF_DATASET="DIS-5K" 

OUTPUT_DIR="exp/grpo"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
RUN_NAME="Seg-R1"
DS_CONFIG="seg-r1/local_scripts/zero1_no_optimizer.json"  # Note that other zero setting would meet bugs related to vllm at current stage.

# Single-GPU friendly defaults: use one visible device and disable vLLM
CUDA_VISIBLE_DEVICES="0" torchrun \
    --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    seg-r1/src/open_r1/grpo_prerl.py \
    --use_vllm false \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --dataset_image datasets/DIS5K/DIS-TR/im \
    --dataset_gt datasets/DIS5K/DIS-TR/gt \
    --max_train_samples 200 \
    --max_prompt_length 2048 \
    --max_completion_length 128 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 200 \
    --save_total_limit 10 \
    --save_only_model true \
    --report_to none \
    --temperature 1.0 \
    --num_generations 2 \
    --sam_device "cpu" \
    --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
    # For single-GPU, you can disable deepspeed to reduce overhead
    # --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
###  --nproc_per_node="6", --vllm_device cuda:6, --sam_device cuda:7