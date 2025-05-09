#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <annotations_path> <image_dir>  <hub_id> [output_dir] [process_count]"
  exit 1
fi

# Assign command-line arguments to variables
ANNOTATIONS_PATH="$1"
IMAGE_DIR="$2"
HUB_ID="$3"


# Run the Python script with the provided arguments
python -m karanta.training.extract_bitmaps_and_create_dataset \
  --annotations_json "$ANNOTATIONS_PATH" \
  --image_dir "$IMAGE_DIR" \
  --hub_dataset_id "$HUB_ID" \

  # Add optional arguments if provided
if [ "$#" -ge 4 ]; then
  OUTPUT_DIR="$4"
  CMD="$CMD --output_dir \"$OUTPUT_DIR\""
fi

if [ "$#" -ge 5 ]; then
  PROCESS_COUNT="$5"
  CMD="$CMD --num_processes \"$PROCESS_COUNT\""
fi

# Execute the command
eval $CMD