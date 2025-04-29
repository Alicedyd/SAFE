#!/bin/bash

# Set your GPU device
GPU_ID=0
CHECKPOINT="./checkpoint/checkpoint-best.pth"
CONFIG="/root/autodl-tmp/code/VAE_RESIZE_AIGC_detection/configs/drct_genimage_chameleon_geneval.yaml"
OUTPUT_DIR="./results/fixed_validation"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run validation with fixed script
echo "Running fixed validation with correct real and fake accuracy metrics..."
CUDA_VISIBLE_DEVICES=$GPU_ID python validate_yaml.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --batch_size 64 \
    --num_workers 8 \
    --input_size 256 \
    --transform_mode 'crop'

echo "Validation complete. Results saved to $OUTPUT_DIR"
echo "Check the transposed CSV file for proper metrics format"