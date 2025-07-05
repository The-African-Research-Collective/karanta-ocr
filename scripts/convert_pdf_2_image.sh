#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <data_path> <output_base_path> <output_format> <process_count>"
  exit 1
fi

# Assign command-line arguments to variables
DATA_PATH="$1"
OUTPUT_BASE_PATH="$2"
OUTPUT_FORMAT="$3"
PROCESS_COUNT="$4"

# Run the Python script with the provided arguments
python -m karanta.data.convert_pdf_2_image \
  --data_path "$DATA_PATH" \
  --output_base_path "$OUTPUT_BASE_PATH" \
  --output_format "$OUTPUT_FORMAT" \
  --num_processes "$PROCESS_COUNT"