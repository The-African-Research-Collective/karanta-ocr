#!/bin/bash

CONFIG=$1

echo "Fine-tuning model with configuration: $CONFIG"

# Pass the YAML file directly as the first argument
python -m karanta.training.run_image_classification $CONFIG