#!/bin/bash

# Celery Workers and Flower Launcher Script
# Usage: ./start_celery.sh --ports "8000,8001,8002,8003" [options]

set -euo pipefail

# Default values
PORTS=""
WORKERS_PER_PORT=3
REDIS_HOST="localhost"
REDIS_PORT=6379
REDIS_PASSWORD=""
FLOWER_PORT=5555
CELERY_APP="workers.celery_app"
LOG_DIR="./celery_logs"
VERBOSE=false
START_FLOWER=true
BROKER_URL=""

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
Celery Workers and Flower Launcher Script

Usage: $0 --ports "8000,8001,8002,8003" [options]

Required Arguments:
  --ports PORTS               Comma-separated list of VLLM ports (e.g., "8000,8001,8002,8003")

Optional Arguments:
  --workers-per-port NUM      Number of workers per port (default: 3)
  --redis-host HOST           Redis host (default: localhost)
  --redis-port PORT           Redis port (default: 6379)
  --redis-password PASS       Redis password (if required)
  --broker-url URL            Full broker URL (overrides redis settings)
  --flower-port PORT          Flower port (default: 5555)
  --celery-app APP            Celery app module (default: workers.celery_app)
  --log-dir DIR               Celery logs directory (default: ./celery_logs)
  --no-flower                 Don't start Flower monitoring
  --verbose                   Enable verbose output
  --help                      Show this help message

Examples:
  # Start workers for 4 VLLM servers
  $0 --ports "8000,8001,8002,8003"
  
  # Start with custom Redis and more workers
  $0 --ports "8000,8001" --workers-per-port 4 --redis-host redis.example.com --redis-port 6380
  
  # Start with Redis password
  $0 --ports "8000,8001,8002,8003" --redis-password mypassword
  
  # Start without Flower
  $0 --ports "8000,8001,8002,8003" --no-flower
  
  # Use custom broker URL
  $0 --ports "8000,8001,8002,8003" --broker-url "redis://user:pass@redis.example.com:6379/0"
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --ports)
                PORTS="$2"
                shift 2
                ;;
            --workers-per-port)
                WORKERS_PER_PORT="$2"
                shift 2
                ;;
            --redis-host)
                REDIS_HOST="$2"
                shift 2
                ;;
            --redis-port)
                REDIS_PORT="$2"
                shift 2
                ;;
            --redis-password)
                REDIS_PASSWORD="$2"
                shift 2
                ;;
            --broker-url)
                BROKER_URL="$2"
                shift 2
                ;;
            --flower-port)
                FLOWER_PORT="$2"
                shift 2
                ;;
            --celery-app)
                CELERY_APP="$2"
                shift 2
                ;;
            --log-dir)
                LOG_DIR="$2"
                shift 2
                ;;
            --no-flower)
                START_FLOWER=false
                shift
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
    if [[ -z "$PORTS" ]]; then
        print_error "Ports are required. Use --ports option."
        exit 1
    fi
    
    # Validate workers per port
    if [[ ! "$WORKERS_PER_PORT" =~ ^[0-9]+$ ]] || [[ "$WORKERS_PER_PORT" -lt 1 ]]; then
        print_error "Workers per port must be a positive integer."
        exit 1
    fi
    
    # Build broker URL if not provided
    if [[ -z "$BROKER_URL" ]]; then
        BROKER_URL="redis://"
        if [[ -n "$REDIS_PASSWORD" ]]; then
            BROKER_URL+="default:${REDIS_PASSWORD}@"
        fi
        BROKER_URL+="${REDIS_HOST}:${REDIS_PORT}/0"
    fi
    
    if [[ $VERBOSE == true ]]; then
        print_info "Broker URL: $BROKER_URL"
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Python
    if ! command_exists python; then
        print_error "Python is not installed or not in PATH."
        exit 1
    fi
    
    # Check Celery
    if ! python -c "import celery" 2>/dev/null; then
        print_error "Celery is not installed. Install with: pip install celery"
        exit 1
    fi
    
    # Check Flower (if needed)
    if [[ $START_FLOWER == true ]]; then
        if ! python -c "import flower" 2>/dev/null; then
            print_error "Flower is not installed. Install with: pip install flower"
            exit 1
        fi
    fi
    
    # Test Redis connection
    print_info "Testing Redis connection..."
    local test_cmd="python -c \"
import redis
try:
    if '$REDIS_PASSWORD':
        r = redis.Redis(host='$REDIS_HOST', port=$REDIS_PORT, password='$REDIS_PASSWORD', decode_responses=True)
    else:
        r = redis.Redis(host='$REDIS_HOST', port=$REDIS_PORT, decode_responses=True)
    r.ping()
    print('Redis connection successful')
except Exception as e:
    print(f'Redis connection failed: {e}')
    exit(1)
\""
    
    if ! eval "$test_cmd"; then
        print_error "Cannot connect to Redis at $REDIS_HOST:$REDIS_PORT"
        print_info "Make sure Redis is running. You can start it with the start_redis.sh script."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Check if port is available
check_port() {
    local port=$1
    local service_name=$2
    
    if ss -tuln | grep -q ":$port "; then
        print_warning "Port $port is already in use (needed for $service_name)."
        return 1
    fi
    
    return 0
}

# Check if Celery app module exists
check_celery_app() {
    print_info "Checking Celery app module: $CELERY_APP"
    
    local test_cmd="python -c \"
try:
    from $CELERY_APP import celery_app
    print('Celery app imported successfully')
except ImportError as e:
    print(f'Failed to import Celery app: {e}')
    exit(1)
except AttributeError as e:
    print(f'Celery app not found in module: {e}')
    exit(1)
\""
    
    if ! eval "$test_cmd"; then
        print_error "Cannot import Celery app from $CELERY_APP"
        print_info "Make sure your Celery app is properly configured and accessible."
        exit 1
    fi
}

# Start Celery worker for a specific port
start_celery_worker() {
    local port=$1
    local worker_index=$2
    local queue_name="gpu_queue_${port}"
    local worker_name="worker_port_${port}_${worker_index}"
    local log_file="$LOG_DIR/${worker_name}.log"
    local pid_file="$LOG_DIR/${worker_name}.pid"
    
    print_info "Starting Celery worker: $worker_name (queue: $queue_name)"
    
    # Build Celery command
    local celery_cmd="celery -A $CELERY_APP worker"
    celery_cmd+=" --hostname=${worker_name}@%h"
    celery_cmd+=" --queues=$queue_name"
    celery_cmd+=" --concurrency=1"
    celery_cmd+=" --loglevel=info"
    celery_cmd+=" --logfile=$log_file"
    celery_cmd+=" --pidfile=$pid_file"
    celery_cmd+=" --detach"

    
    if [[ $VERBOSE == true ]]; then
        print_info "Executing: $celery_cmd"
    fi
    
    # Start worker
    if eval "$celery_cmd"; then
        # Wait a moment for PID file to be created
        sleep 1
        
        # Verify worker started
        if [[ -f "$pid_file" ]]; then
            local pid
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                print_success "Worker $worker_name started (PID: $pid)"
                return 0
            fi
        fi
    fi
    
    print_error "Failed to start worker $worker_name"
    return 1
}

# Start all Celery workers
start_celery_workers() {
    print_info "Starting Celery workers..."
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Convert comma-separated ports to array
    IFS=',' read -ra PORT_ARRAY <<< "$PORTS"
    
    local failed_workers=()
    local successful_workers=()
    
    # Start workers for each port
    for port in "${PORT_ARRAY[@]}"; do
        for ((i=1; i<=WORKERS_PER_PORT; i++)); do
            if start_celery_worker "$port" "$i"; then
                successful_workers+=("worker_port_${port}_${i}")
            else
                failed_workers+=("worker_port_${port}_${i}")
                # Continue with other workers even if one fails
            fi
            
            # Small delay between worker starts
            sleep 0.5
        done
    done
    
    # Report results
    if [[ ${#successful_workers[@]} -gt 0 ]]; then
        print_success "Successfully started ${#successful_workers[@]} workers"
    fi
    
    if [[ ${#failed_workers[@]} -gt 0 ]]; then
        print_error "Failed to start ${#failed_workers[@]} workers:"
        for worker in "${failed_workers[@]}"; do
            print_error "  ✗ $worker"
        done
        
        # Don't exit if some workers started successfully
        if [[ ${#successful_workers[@]} -eq 0 ]]; then
            print_error "No workers started successfully"
            return 1
        else
            print_warning "Some workers failed to start, but continuing with ${#successful_workers[@]} workers"
        fi
    fi
    
    return 0
}

# Start Flower monitoring
start_flower() {
    if [[ $START_FLOWER == false ]]; then
        print_info "Skipping Flower startup (--no-flower flag)"
        return 0
    fi
    
    print_info "Starting Flower monitoring..."
    
    # Check if port is available
    if ! check_port "$FLOWER_PORT" "Flower"; then
        print_warning "Port $FLOWER_PORT might already be in use"
        # Try to check if Flower is already running
        if curl -sf "http://localhost:$FLOWER_PORT" &>/dev/null; then
            print_success "Flower already running on http://localhost:$FLOWER_PORT"
            return 0
        fi
    fi
    
    local flower_log="$LOG_DIR/flower.log"
    local flower_pid_file="$LOG_DIR/flower.pid"
    
    # Build Flower command
    local flower_cmd="celery -A $CELERY_APP flower"
    flower_cmd+=" --port=$FLOWER_PORT"
    flower_cmd+=" --broker='$BROKER_URL'"
    
    if [[ $VERBOSE == true ]]; then
        print_info "Executing: $flower_cmd"
    fi
    
    # Start Flower in background
    nohup $flower_cmd > "$flower_log" 2>&1 &
    local pid=$!
    
    # Save PID
    echo "$pid" > "$flower_pid_file"
    
    # Wait for Flower to start
    local attempts=0
    while [[ $attempts -lt 30 ]]; do
        if curl -sf "http://localhost:$FLOWER_PORT" &>/dev/null; then
            print_success "Flower started successfully (http://localhost:$FLOWER_PORT)"
            return 0
        fi
        
        sleep 1
        ((attempts++))
    done
    
    print_warning "Flower might be starting slowly. Check http://localhost:$FLOWER_PORT"
    return 0
}

# Create cleanup script
create_cleanup_script() {
    local cleanup_script="$LOG_DIR/cleanup_celery_system.sh"
    
    cat > "$cleanup_script" << 'EOF'
#!/bin/bash
# Auto-generated cleanup script for Celery system

SCRIPT_DIR=$(dirname "$0")
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_info "Stopping Celery system..."

# Stop Celery workers
for pid_file in "$SCRIPT_DIR"/worker_*.pid; do
    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        worker_name=$(basename "$pid_file" .pid)
        
        if kill -0 "$pid" 2>/dev/null; then
            print_info "Stopping $worker_name (PID: $pid)"
            kill -TERM "$pid"
            
            # Wait for graceful shutdown
            sleep 3
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                print_info "Force killing $worker_name (PID: $pid)"
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        
        rm -f "$pid_file"
    fi
done

# Stop Flower
if [[ -f "$SCRIPT_DIR/flower.pid" ]]; then
    pid=$(cat "$SCRIPT_DIR/flower.pid")
    if kill -0 "$pid" 2>/dev/null; then
        print_info "Stopping Flower (PID: $pid)"
        kill "$pid" 2>/dev/null || true
    fi
    rm -f "$SCRIPT_DIR/flower.pid"
fi

print_info "Celery system stopped"
EOF
    
    chmod +x "$cleanup_script"
    print_info "Cleanup script created: $cleanup_script"
}

# Create system summary
create_system_summary() {
    local summary_file="$LOG_DIR/system_summary.json"
    
    # Convert comma-separated ports to array
    IFS=',' read -ra PORT_ARRAY <<< "$PORTS"
    
    cat > "$summary_file" << EOF
{
    "launch_time": "$(date -Iseconds)",
    "broker_url": "$BROKER_URL",
    "flower": {
        "enabled": $START_FLOWER,
        "port": $FLOWER_PORT,
        "url": "http://localhost:$FLOWER_PORT"
    },
    "workers": {
        "per_port": $WORKERS_PER_PORT,
        "total": $((${#PORT_ARRAY[@]} * WORKERS_PER_PORT)),
        "celery_app": "$CELERY_APP"
    },
    "port_mapping": [
EOF
    
    local first=true
    for port in "${PORT_ARRAY[@]}"; do
        if [[ $first == true ]]; then
            first=false
        else
            echo "," >> "$summary_file"
        fi
        
        cat >> "$summary_file" << EOF
        {
            "vllm_port": $port,
            "queue": "gpu_queue_${port}",
            "workers": [
EOF
        
        for ((i=1; i<=WORKERS_PER_PORT; i++)); do
            if [[ $i -gt 1 ]]; then
                echo "," >> "$summary_file"
            fi
            cat >> "$summary_file" << EOF
                {
                    "name": "worker_port_${port}_${i}",
                    "log_file": "$LOG_DIR/worker_port_${port}_${i}.log"
                }
EOF
        done
        
        cat >> "$summary_file" << EOF
            ]
        }
EOF
    done
    
    cat >> "$summary_file" << EOF
    ]
}
EOF
    
    print_info "System summary created: $summary_file"
}

# Main function
main() {
    print_info "Celery Workers and Flower Launcher Starting..."
    
    # Parse arguments
    parse_args "$@"
    
    # Validate arguments
    validate_args
    
    # Check prerequisites
    check_prerequisites
    
    # Check Celery app
    check_celery_app
    
    # Start Celery workers
    if ! start_celery_workers; then
        print_error "Failed to start some Celery workers"
        exit 1
    fi
    
    # Start Flower
    start_flower
    
    # Create cleanup script and summary
    create_cleanup_script
    create_system_summary
    
    # Final summary
    echo
    print_success "=== Celery System Launch Summary ==="
    
    IFS=',' read -ra PORT_ARRAY <<< "$PORTS"
    local total_workers=$((${#PORT_ARRAY[@]} * WORKERS_PER_PORT))
    print_success "✓ $total_workers Celery workers started across ${#PORT_ARRAY[@]} VLLM ports"
    
    if [[ $START_FLOWER == true ]]; then
        print_success "✓ Flower monitoring available at http://localhost:$FLOWER_PORT"
    fi
    
    echo
    print_info "Broker URL: $BROKER_URL"
    print_info "Logs directory: $LOG_DIR"
    print_info "To stop all services: $LOG_DIR/cleanup_celery_system.sh"
    
    echo
    print_info "Queue configuration:"
    for port in "${PORT_ARRAY[@]}"; do
        print_info "  VLLM port $port: queue 'gpu_queue_${port}' with $WORKERS_PER_PORT workers"
    done
}

# Handle Ctrl+C gracefully
trap 'print_warning "Received interrupt signal. Run $LOG_DIR/cleanup_celery_system.sh to stop services."; exit 130' INT

# Run main function
main "$@"