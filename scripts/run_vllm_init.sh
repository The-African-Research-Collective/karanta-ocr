# Triton==3.1.0 &&

export VLLM_PORT=$1
export VLLM_MODEL=$2 #
export CUDA_VISIBLE_DEVICES=0

# Set up Python virtual environment
echo "Setting up Python virtual environment..."

source .venv/bin/activate

# Configure and start the vLLM server
echo "Starting vLLM server..."
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

mkdir -p logs/.config/vllm && touch logs/.config/vllm/do_not_track

# Check if required environment variables are set
if [ -z "$VLLM_PORT" ] || [ -z "$VLLM_MODEL" ]; then
    echo "Error: Required environment variables are not set."
    exit 1
fi

# Start the server in a new tmux session
tmux new -s vllm -d "python -m vllm.entrypoints.openai.api_server \
    --model $VLLM_MODEL \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --port $VLLM_PORT" \
    --api-key "karanta-ocr" \
    --task "generate"

echo "vLLM server started successfully in tmux session 'vllm'."