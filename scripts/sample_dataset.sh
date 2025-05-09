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
DATASET_NAME=$(grep '^dataset_name:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
SPLIT=$(grep '^split:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
COLUMNS_TO_KEEP=$(grep '^columns_to_keep:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
RENAME_COLUMNS=$(grep '^rename_columns:' "$CONFIG_FILE" | awk -F': ' '{print $2}' | tr -d '"')
HUB_ID=$(grep '^hub_id:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
PRIVATE=$(grep '^private:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')

# Run the Python script with the extracted arguments
python -m karanta.data.sample_existing_dataset \
  --dataset_name "$DATASET_NAME" \
  --split "$SPLIT" \
  --columns_to_keep "$COLUMNS_TO_KEEP" \
  --rename_columns "$RENAME_COLUMNS" \
  --hub_id "$HUB_ID" \
  --private "$PRIVATE"