#!/bin/bash

# This script processes batches of requests using the vLLM framework.

MODEL=$1
PORT=$2
PROCESSING_FILE=$3
INPUT_DIR=$4
OUTPUT_DIR=$5
START_INDEX=$6
END_INDEX=$7

# Loop through numbers 1-10
for i in $(seq $START_INDEX $END_INDEX); do
    echo "Processing batch $i..."
    
    python3 -m karanta.data.groundtruth.process_vllm_requests_distributed \
        ${INPUT_DIR}/batch_llm_requsts_file_allenai_olmOCR_7B_0225_preview_${i}.jsonl \
        --results-file ${OUTPUT_DIR}/batch_llm_requsts_file_allenai_olmOCR_7B_0225_preview_${i}.jsonl \
        --max-concurrent-per-server 2 \
        --total-concurrent-limit 8 \
        --servers $PORT \
        --load-balancing adaptive_queue \
        --model "$MODEL" \
        --progress-file ${PROCESSING_FILE}
    
    echo "Completed batch $i"
done

echo "All batches processed!"