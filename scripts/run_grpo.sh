#!/bin/bash


# The latest vllm==0.7.3 is required for this script: pip3 install vllm==0.7.3
# The latest transformers is required too, install by: pip install git+https://github.com/huggingface/transformers.git@a40f1ac602fe900281722254c52ce3773f28eb0e



export DEBUG_MODE="true"
export LOG_PATH="exp/vllm_run.txt"
QWEN_PATH="/path/to/prerl/ckpt"
HF_DATASET="COD" 

OUTPUT_DIR="exp/grpo"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
RUN_NAME="Seg-R1"
DS_CONFIG="seg-r1/local_scripts/zero1_no_optimizer.json"  # Note that other zero setting would meet bugs related to vllm at current stage.

# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
    --nproc_per_node="6" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    seg-r1/src/open_r1/grpo.py \
    --use_vllm true \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --dataset_image datasets/COD10K-v3/Train/Image,datasets/CAMO-V.1.0-CVIU2019/Images/Train \
    --dataset_gt datasets/COD10K-v3/Train/GT_Object,datasets/CAMO-V.1.0-CVIU2019/GT/Train \
    --max_prompt_length 6666 \
    --max_completion_length 400 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 200 \
    --save_total_limit 10 \
    --save_only_model true \
    --report_to wandb \
    --temperature 1.0 \
    --num_generations 4 \
    --vllm_device "cuda:6" \
    --sam_device "cuda:7" \
    --vllm_gpu_memory_utilization 0.7 \
    --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
