#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yaml" ]; then
    print_error "docker-compose.yaml not found in current directory!"
    exit 1
fi

# Request experiment name
while true; do
    read -p "Enter experiment name: " EXPERIMENT_NAME
    
    # Check if experiment name is not empty
    if [ -n "$EXPERIMENT_NAME" ]; then
        # Remove spaces and special characters, replace with underscores
        EXPERIMENT_NAME=$(echo "$EXPERIMENT_NAME" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/__*/_/g' | sed 's/^_\|_$//g')
        break
    else
        print_warning "Experiment name cannot be empty. Please try again."
    fi
done

# Generate run_id (datetime + uuid)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
UUID_PART=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid 2>/dev/null || openssl rand -hex 4)
UUID_SHORT=$(echo "$UUID_PART" | cut -d'-' -f1)
RUN_ID="${TIMESTAMP}_${UUID_SHORT}"

print_info "Experiment Name: $EXPERIMENT_NAME"
print_info "Generated Run ID: $RUN_ID"


print_info "Environment variables written to .env file"

# Export variables for current session
export EXPERIMENT_NAME="$EXPERIMENT_NAME"
export RUN_ID="$RUN_ID"
export TIMESTAMP="$TIMESTAMP"
export LOG_FILE="/output/test_log.tsv"
export ENABLE_CONSOLE_LOGS=true

export ENABLE_RABBITMQ="true"
export RABBITMQ_HOST="192.168.88.253"
export RABBITMQ_USERNAME="tech"
export RABBITMQ_PASSWORD="tech"

export RABBITMQ_EXCHANGE=""                    # Use default exchange
export RABBITMQ_ROUTING_KEY="rabbitmq_rl_snake_logs"     # Same as your queue name

print_info "Starting Docker Compose with --build flag..."

# Run docker-compose up --build
if docker compose up --build; then
    print_info "Docker Compose completed successfully"
else
    print_error "Docker Compose failed with exit code $?"
    exit 1
fi
