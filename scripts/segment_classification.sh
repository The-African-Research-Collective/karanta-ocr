#!/bin/bash

# Script to run image classification fine-tuining and evaluation using the provided configuration.

CONFIG=$1
NUM_GPUS=$2
TRAINING_PRECISION=$3


echo "Fine-tuning model using $NUM_GPUS GPUs"


# Check that precision exists among the available list of options ("fp32", "bf16", "fp16")
if [[ $TRAINING_PRECISION != "fp32" && $TRAINING_PRECISION != "bf16" && $TRAINING_PRECISION != "fp16" ]]; then
    echo "Invalid training precision. Please choose from 'fp32', 'bf16', or 'fp16'."
    exit 1
fi

# Set CUDA devices (update based on your system's GPU IDs)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

# Run the image classification script with the specified configuration
accelerate launch \
    --mixed_precision $TRAINING_PRECISION \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    ./karanta/training/run_image_classification.py $CONFIG