#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <data_path> <output_base_path> <output_format>"
  exit 1
fi

# Assign command-line arguments to variables
DATA_PATH="$1"
OUTPUT_BASE_PATH="$2"
OUTPUT_FORMAT="$3"

# Run the Python script with the provided arguments
python karanta/data/pdf2_image.py \
  --data_path "$DATA_PATH" \
  --output_base_path "$OUTPUT_BASE_PATH" \
  --output_format "$OUTPUT_FORMAT"