#!/bin/bash

# Check if the configuration file is provided as an argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <config_file>"
  exit 1
fi

CONFIG_FILE="$1"

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Configuration file '$CONFIG_FILE' not found."
  exit 1
fi

# Extract values from the YAML file
DATA_PATH=$(grep '^data_path:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
OUTPUT_BASE_PATH=$(grep '^output_base_path:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
OUTPUT_FORMAT=$(grep '^output_format:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')

# Run the Python script with the extracted arguments
python karanta/data/pdf2_image.py \
  --data_path "$DATA_PATH" \
  --output_base_path "$OUTPUT_BASE_PATH" \
  --output_format "$OUTPUT_FORMAT" 