export VLLM_PORT=$1
export VLLM_MODEL=$2


# Set up Python virtual environment
echo "Setting up Python virtual environment..."

uv venv  || { echo "Failed to create Python virtual environment"; exit 1; }

source .venv/bin/activate
pip install vllm || { echo "Failed to install vllm"; exit 1; }

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
    --dtype auto \
    --gpu-memory-utilization 0.9 \
    --port $VLLM_PORT"

echo "vLLM server started successfully in tmux session 'vllm'."