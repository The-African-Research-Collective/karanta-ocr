#!/bin/bash

CONFIG=$1

echo "Fine-tuning model with configuration: $CONFIG"

# Run the image classification script with the specified configuration
python karanta/training/run_image_classification.py --config $CONFIG