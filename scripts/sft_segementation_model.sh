#!/bin/bash

CONFIG=$1

echo "Fine-tuning segmentation model with configuration: $CONFIG"

# Pass the YAML file directly as the first argument
python karanta/training/sft_segmentation.py $CONFIG