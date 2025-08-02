#!/bin/bash

# Loop through numbers 1-10
for i in {44..132}; do
    echo "Processing batch $i..."
    
    python3 -m karanta.data.process_vllm_requests_distributed \
        data/jw/requests_dir/batch_llm_requsts_file_allenai_olmOCR_7B_0225_preview_${i}.jsonl \
        --results-file data/jw/response_dir/batch_llm_requsts_file_allenai_olmOCR_7B_0225_preview_${i}.jsonl \
        --max-concurrent-per-server 2 \
        --total-concurrent-limit 8 \
        --servers 8005 8006 \
        --load-balancing adaptive_queue
    
    echo "Completed batch $i"
done

echo "All batches processed!"