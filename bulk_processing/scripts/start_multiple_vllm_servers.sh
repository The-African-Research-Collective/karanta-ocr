#!/bin/bash

# VLLM Server Launcher Script
# Usage: ./start_multiple_vllm_servers.sh --gpus "0,1,2,3" --ports "8000,8001,8002,8003" --model "meta-llama/Llama-2-7b-hf" [options]

set -euo pipefail

# Default values
MODEL=""
GPUS=""
PORTS=""
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=""
TRUST_REMOTE_CODE=true
DTYPE="bfloat16"
MAX_NUM_SEQS=256
HEALTH_CHECK_TIMEOUT=300
LOG_DIR="./vllm_logs"
VERBOSE=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to show usage
show_usage() {
    cat << EOF
VLLM Server Launcher Script

Usage: $0 --gpus "0,1,2,3" --ports "8000,8001,8002,8003" --model "model_name" [options]

Required Arguments:
  --gpus GPUS                 Comma-separated list of GPU IDs (e.g., "0,1,2,3")
  --ports PORTS               Comma-separated list of ports (e.g., "8000,8001,8002,8003")
  --model MODEL               Model name or path (e.g., "meta-llama/Llama-2-7b-hf")

Optional Arguments:
  --tensor-parallel-size SIZE Tensor parallel size (default: 1)
  --max-model-len LENGTH      Maximum model length
  --trust-remote-code         Trust remote code (default: false)
  --dtype DTYPE               Data type (default: auto)
  --max-num-seqs SEQS         Maximum number of sequences (default: 256)
  --timeout SECONDS           Health check timeout in seconds (default: 300)
  --log-dir DIR               Log directory (default: ./vllm_logs)
  --verbose                   Enable verbose output
  --help                      Show this help message

Examples:
  # Launch on 4 GPUs with basic settings
  $0 --gpus "0,1,2,3" --ports "8000,8001,8002,8003" --model "meta-llama/Llama-2-7b-hf"
  
  # Launch with custom settings
  $0 --gpus "0,1" --ports "8000,8001" --model "meta-llama/Llama-2-13b-hf" \\
     --tensor-parallel-size 2 --max-model-len 4096 --trust-remote-code
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpus)
                GPUS="$2"
                shift 2
                ;;
            --ports)
                PORTS="$2"
                shift 2
                ;;
            --model)
                MODEL="$2"
                shift 2
                ;;
            --tensor-parallel-size)
                TENSOR_PARALLEL_SIZE="$2"
                shift 2
                ;;
            --max-model-len)
                MAX_MODEL_LEN="$2"
                shift 2
                ;;
            --trust-remote-code)
                TRUST_REMOTE_CODE=true
                shift
                ;;
            --dtype)
                DTYPE="$2"
                shift 2
                ;;
            --max-num-seqs)
                MAX_NUM_SEQS="$2"
                shift 2
                ;;
            --timeout)
                HEALTH_CHECK_TIMEOUT="$2"
                shift 2
                ;;
            --log-dir)
                LOG_DIR="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Validate required arguments
validate_args() {
    if [[ -z "$GPUS" ]]; then
        print_error "GPU IDs are required. Use --gpus option."
        exit 1
    fi
    
    if [[ -z "$PORTS" ]]; then
        print_error "Ports are required. Use --ports option."
        exit 1
    fi
    
    if [[ -z "$MODEL" ]]; then
        print_error "Model name is required. Use --model option."
        exit 1
    fi
}

# Check if nvidia-smi is available
check_nvidia_smi() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        exit 1
    fi
}

# Check if GPU exists and is available
check_gpu() {
    local gpu_id=$1
    
    if ! nvidia-smi -i "$gpu_id" &> /dev/null; then
        print_error "GPU $gpu_id not found or not accessible."
        return 1
    fi
    
    # Check if GPU is being used heavily (>90% utilization)
    local gpu_util
    gpu_util=$(nvidia-smi -i "$gpu_id" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    
    if [[ $gpu_util -gt 90 ]]; then
        print_warning "GPU $gpu_id is heavily utilized ($gpu_util%). Consider using a different GPU."
    fi
    
    print_info "GPU $gpu_id is available (utilization: $gpu_util%)"
    return 0
}

# Check if port is available
check_port() {
    local port=$1
    
    if ss -tuln | grep -q ":$port "; then
        print_error "Port $port is already in use."
        return 1
    fi
    
    print_info "Port $port is available"
    return 0
}

# Check if Python package is installed
check_python_package() {
    local package=$1
    
    if ! python -c "import $package" &> /dev/null; then
        print_error "Python package '$package' is not installed."
        return 1
    fi
    
    return 0
}

# Check all prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check nvidia-smi
    check_nvidia_smi
    
    # Check Python packages
    check_python_package "vllm" || exit 1
    
    # Convert comma-separated strings to arrays
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    IFS=',' read -ra PORT_ARRAY <<< "$PORTS"
    
    # Check if number of GPUs matches number of ports
    if [[ ${#GPU_ARRAY[@]} -ne ${#PORT_ARRAY[@]} ]]; then
        print_error "Number of GPUs (${#GPU_ARRAY[@]}) must match number of ports (${#PORT_ARRAY[@]})"
        exit 1
    fi
    
    # Check each GPU
    for gpu_id in "${GPU_ARRAY[@]}"; do
        if ! check_gpu "$gpu_id"; then
            exit 1
        fi
    done
    
    # Check each port
    for port in "${PORT_ARRAY[@]}"; do
        if ! check_port "$port"; then
            exit 1
        fi
    done
    
    print_success "All prerequisites checked successfully"
}

# Wait for server to be ready
wait_for_server() {
    local port=$1
    local gpu_id=$2
    local timeout=$3
    local start_time
    start_time=$(date +%s)
    
    print_info "Waiting for VLLM server on GPU $gpu_id (port $port) to be ready..."
    
    while true; do
        local current_time
        current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [[ $elapsed -gt $timeout ]]; then
            print_error "Timeout waiting for server on port $port (GPU $gpu_id)"
            return 1
        fi
        
        # Check if server responds to health check
        if curl -sf "http://localhost:$port/health" &> /dev/null; then
            print_success "VLLM server on GPU $gpu_id (port $port) is ready!"
            return 0
        fi
        
        if [[ $VERBOSE == true ]]; then
            print_info "Waiting for server on port $port... (${elapsed}s elapsed)"
        fi
        
        sleep 10
    done
}

# Launch a single VLLM server
launch_vllm_server() {
    local gpu_id=$1
    local port=$2
    local log_file="$LOG_DIR/vllm_gpu_${gpu_id}_port_${port}.log"
    
    # Create log directory if it doesn't exist
    mkdir -p "$LOG_DIR"
    
    print_info "Launching VLLM server on GPU $gpu_id, port $port..."
    
    # Build command
    local cmd="CUDA_VISIBLE_DEVICES=$gpu_id python -m vllm.entrypoints.openai.api_server"
    cmd+=" --model $MODEL"
    cmd+=" --port $port"
    cmd+=" --dtype $DTYPE"
    
    if [[ -n "$MAX_MODEL_LEN" ]]; then
        cmd+=" --max-model-len $MAX_MODEL_LEN"
    fi
    
    if [[ $TRUST_REMOTE_CODE == true ]]; then
        cmd+=" --trust-remote-code"
    fi
    
    if [[ $VERBOSE == true ]]; then
        print_info "Executing: $cmd"
    fi
    
    # Launch server in background
    eval "$cmd" > "$log_file" 2>&1 &
    local pid=$!
    
    # Store PID for cleanup
    echo "$pid" > "$LOG_DIR/vllm_gpu_${gpu_id}_port_${port}.pid"
    
    print_info "VLLM server started on GPU $gpu_id (PID: $pid, Log: $log_file)"
    
    # Wait for server to be ready
    if wait_for_server "$port" "$gpu_id" "$HEALTH_CHECK_TIMEOUT"; then
        return 0
    else
        # Kill the process if it failed to start
        kill "$pid" 2>/dev/null || true
        rm -f "$LOG_DIR/vllm_gpu_${gpu_id}_port_${port}.pid"
        return 1
    fi
}

# Create cleanup script
create_cleanup_script() {
    local cleanup_script="$LOG_DIR/cleanup_servers.sh"
    
    cat > "$cleanup_script" << 'EOF'
#!/bin/bash
# Auto-generated cleanup script for VLLM servers

LOG_DIR=$(dirname "$0")
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_info "Stopping VLLM servers..."

# Find all PID files
for pid_file in "$LOG_DIR"/*.pid; do
    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        server_name=$(basename "$pid_file" .pid)
        
        if kill -9 "$pid" 2>/dev/null; then
            print_info "Stopping $server_name (PID: $pid)"
            kill "$pid"
            
            # Wait for graceful shutdown
            sleep 2
            
            # Force kill if still running
            if kill -9 "$pid" 2>/dev/null; then
                print_info "Force killing $server_name (PID: $pid)"
                kill -9 "$pid" 2>/dev/null || true
            fi
        else
            print_info "$server_name was already stopped"
        fi
        
        rm -f "$pid_file"
    fi
done

print_info "All VLLM servers stopped"
EOF
    
    chmod +x "$cleanup_script"
    print_info "Cleanup script created: $cleanup_script"
}

# Create server info summary
create_server_summary() {
    local summary_file="$LOG_DIR/server_summary.json"
    
    cat > "$summary_file" << EOF
{
    "launch_time": "$(date -Iseconds)",
    "model": "$MODEL",
    "servers": [
EOF
    
    local first=true
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    IFS=',' read -ra PORT_ARRAY <<< "$PORTS"
    
    for i in "${!GPU_ARRAY[@]}"; do
        local gpu_id="${GPU_ARRAY[i]}"
        local port="${PORT_ARRAY[i]}"
        
        if [[ $first == true ]]; then
            first=false
        else
            echo "," >> "$summary_file"
        fi
        
        cat >> "$summary_file" << EOF
        {
            "gpu_id": $gpu_id,
            "port": $port,
            "url": "http://localhost:$port",
            "health_url": "http://localhost:$port/health",
            "log_file": "$LOG_DIR/vllm_gpu_${gpu_id}_port_${port}.log"
        }
EOF
    done
    
    cat >> "$summary_file" << EOF

    ],
    "config": {
        "tensor_parallel_size": $TENSOR_PARALLEL_SIZE,
        "dtype": "$DTYPE",
        "max_num_seqs": $MAX_NUM_SEQS,
        "max_model_len": "${MAX_MODEL_LEN:-"auto"}",
        "trust_remote_code": $TRUST_REMOTE_CODE
    }
}
EOF
    
    print_info "Server summary created: $summary_file"
}

# Main function
main() {
    print_info "VLLM Server Launcher Starting..."
    
    # Parse arguments
    parse_args "$@"
    
    # Validate arguments
    validate_args
    
    # Check prerequisites
    check_prerequisites
    
    # Convert comma-separated strings to arrays
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    IFS=',' read -ra PORT_ARRAY <<< "$PORTS"
    
    # Launch servers
    local failed_servers=()
    local successful_servers=()
    
    for i in "${!GPU_ARRAY[@]}"; do
        local gpu_id="${GPU_ARRAY[i]}"
        local port="${PORT_ARRAY[i]}"
        
        if launch_vllm_server "$gpu_id" "$port"; then
            successful_servers+=("GPU $gpu_id:$port")
        else
            failed_servers+=("GPU $gpu_id:$port")
        fi
    done
    
    # Create cleanup script and summary
    create_cleanup_script
    create_server_summary
    
    # Report results
    echo
    print_success "=== VLLM Server Launch Summary ==="
    
    if [[ ${#successful_servers[@]} -gt 0 ]]; then
        print_success "Successfully launched servers:"
        for server in "${successful_servers[@]}"; do
            print_success "  ✓ $server"
        done
    fi
    
    if [[ ${#failed_servers[@]} -gt 0 ]]; then
        print_error "Failed to launch servers:"
        for server in "${failed_servers[@]}"; do
            print_error "  ✗ $server"
        done
        echo
        print_info "Check log files in $LOG_DIR for error details"
        exit 1
    fi
    
    echo
    print_success "All VLLM servers are ready!"
    print_info "Model: $MODEL"
    print_info "Logs directory: $LOG_DIR"
    print_info "To stop all servers: $LOG_DIR/cleanup_servers.sh"
    
    # Show server URLs
    echo
    print_info "Server URLs:"
    for i in "${!GPU_ARRAY[@]}"; do
        local gpu_id="${GPU_ARRAY[i]}"
        local port="${PORT_ARRAY[i]}"
        print_info "  GPU $gpu_id: http://localhost:$port"
    done
}

# Handle Ctrl+C gracefully
trap 'print_warning "Received interrupt signal. Run $LOG_DIR/cleanup_servers.sh to stop servers."; exit 130' INT

# Run main function
main "$@"