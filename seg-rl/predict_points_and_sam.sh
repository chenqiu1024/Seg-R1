#!/bin/bash
## Example: seg-rl/predict_points_and_sam.sh pred_251022 12 pretrained/points_predictor-251001-160epochs.pt cuda
# Check if correct number of arguments are provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <name_of_task> <length_points_sequence> <path_to_model> <device>"
    echo "Example: $0 \"pred_251015\" 12 \"outputs/braintumour/points_predictor-251001-50epochs.pt\" \"cuda\""
    exit 1
fi

# Get the arguments
task_name="$1"
n_points="$2"
path_to_model="$3"
device="$4"
# Validate that the second argument is a positive integer
if ! [[ "$n_points" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Length of points sequence must be a positive integer"
    echo "Usage: $0 <name_of_task> <length_points_sequence> <path_to_model> <device>"
    exit 1
fi

# Initialize conda and activate the seg-r1 environment
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate seg-r1

# Set the Python path explicitly
# PYTHON_PATH="/opt/anaconda3/envs/seg-r1/bin/python"
PYTHON_PATH="/root/autodl-tmp/envs/seg-r1/bin/python"

# Verify we're using the correct Python
echo "Using python: $PYTHON_PATH"
echo "Using python version: $($PYTHON_PATH --version)"

# Check for existing files and folders
JSONL_FILE="outputs/braintumour/$task_name.jsonl"
SAM_MASKS_DIR="outputs/braintumour/${task_name}_sam_masks"
DBG_PRED_POINTS_DIR="outputs/braintumour/${task_name}_dbg_pred_points"

# Control whether to skip initial first-prompt generation when keeping existing JSONL
SKIP_FIRST=0

existing_files=()
if [ -f "$JSONL_FILE" ]; then
    existing_files+=("$JSONL_FILE")
fi
if [ -d "$SAM_MASKS_DIR" ]; then
    existing_files+=("$SAM_MASKS_DIR")
fi
if [ -d "$DBG_PRED_POINTS_DIR" ]; then
    existing_files+=("$DBG_PRED_POINTS_DIR")
fi

if [ ${#existing_files[@]} -gt 0 ]; then
    echo ""
    echo "Warning: The following files/folders already exist:"
    for file in "${existing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    read -p "Do you want to delete these existing files/folders and continue? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing files/folders..."
        for file in "${existing_files[@]}"; do
            if [ -f "$file" ]; then
                rm -f "$file"
                echo "  Deleted file: $file"
            elif [ -d "$file" ]; then
                rm -rf "$file"
                echo "  Deleted directory: $file"
            fi
        done
        echo "Cleanup completed. Proceeding with execution..."
    else
        echo "Keeping existing files/folders. Proceeding without deletion..."
        # If the JSONL already exists, skip the initial first-prompt generation block
        if [ -f "$JSONL_FILE" ]; then
            SKIP_FIRST=1
            echo "Detected existing JSONL ($JSONL_FILE). Will skip initial first-prompt generation."
        fi
    fi
fi

# Print the arguments
echo "Name of task: $task_name"
echo "Length of points sequence: $n_points"
echo "Path to model: $path_to_model"
if [ "$SKIP_FIRST" -eq 0 ]; then
    echo "Generate the first point prompt for each image:"

    $PYTHON_PATH -m seg-rl.heatmap.predict_next_point_from_model \
        --model_path $path_to_model \
        --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
        --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
        --output_json $JSONL_FILE

    $PYTHON_PATH seg-rl/sam2_segment_from_points.py \
        --input_jsonl $JSONL_FILE \
        --json_output $JSONL_FILE \
        --output_dir $SAM_MASKS_DIR \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --device $device \
        --resize 512 512 \
        --skip_existing

    $PYTHON_PATH seg-rl/evaluation/eval_sam_masks.py --input_json $JSONL_FILE --num_prompts 1
else
    echo "Skipping initial first-prompt generation and starting from the iterative loop..."
fi
# Loop from 1 to (n_points - 1)
echo "Starting loop from 1 to $((n_points - 1)):"
for i in $(seq 1 $((n_points - 1))); do
    echo "Generate the $((i+1))th point prompt for each image"
    # Generate the next point prompt
    $PYTHON_PATH -m seg-rl.heatmap.predict_next_point_from_model \
        --model_path $path_to_model \
        --appendto_json $JSONL_FILE \
        --sam_dir $SAM_MASKS_DIR 
    
    $PYTHON_PATH seg-rl/sam2_segment_from_points.py \
      --input_jsonl $JSONL_FILE \
      --json_output $JSONL_FILE \
      --output_dir $SAM_MASKS_DIR \
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device $device \
      --resize 512 512 \
      --skip_existing

    $PYTHON_PATH seg-rl/evaluation/eval_sam_masks.py --input_json $JSONL_FILE --num_prompts $((i+1))
done

echo "Evaluate the performance of the model"

$PYTHON_PATH seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --debug_json $JSONL_FILE \
    --debug_output_dir $DBG_PRED_POINTS_DIR

$PYTHON_PATH seg-rl/sam2_automatic_evaluation.py \
    --input_jsonl $JSONL_FILE \
    --sam_masks_dir $SAM_MASKS_DIR \
    --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
    --device cuda

