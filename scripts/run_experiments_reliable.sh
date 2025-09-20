#!/bin/bash
set -euo pipefail

# Reliable experiment runner for ENV Service + Inference Job architecture
BASE_VALUES="k8s/snake-rl/values.yaml"
CHART_DIR="k8s/snake-rl"
NAMESPACE="experiments"
ENV_RELEASE="snake-env"

# Experiment configuration
grid_sizes=(7 9 11)
beta_values=(0.0001 0.001 0.05)
learning_rates=(0.001 0.01)

# Infrastructure settings
TIMEOUT_SECONDS=$((30*60))  # 30min timeout per experiment
TARGET_EPISODES=100         # Episodes per experiment
POLL_SECONDS=10

# ClickHouse connection (with fallback detection)
CH_NAMESPACE="data-stack"
CH_USER="default"
CH_PASS=""  # Will be auto-detected

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"; }

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -f override-*.yaml
    if [[ "${CLEANUP_ENV:-false}" == "true" ]]; then
        log "Cleaning up environment service..."
        helm uninstall "$ENV_RELEASE" --namespace "$NAMESPACE" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Auto-detect ClickHouse connection
setup_clickhouse() {
    log "Setting up ClickHouse connection..."
    
    # Auto-detect ClickHouse pod
    CH_POD=$(kubectl get pods -n "$CH_NAMESPACE" -l app.kubernetes.io/name=clickhouse -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [[ -z "$CH_POD" ]]; then
        error "ClickHouse pod not found in namespace '$CH_NAMESPACE'"
        return 1
    fi
    
    # Auto-detect password
    if [[ -z "$CH_PASS" ]]; then
        CH_PASS=$(kubectl get secret clickhouse-credentials -n "$CH_NAMESPACE" -o jsonpath="{.data.password}" 2>/dev/null | base64 -d || echo "")
        if [[ -z "$CH_PASS" ]]; then
            warn "Could not auto-detect ClickHouse password, using empty password"
        fi
    fi
    
    log "Using ClickHouse pod: $CH_POD"
}

# Wait for Kubernetes job completion with proper monitoring
wait_for_job_completion() {
    local job_name=$1
    local exp_name=$2
    local start_ts=$(date +%s)
    
    log "Monitoring job: $job_name (experiment: $exp_name)"
    
    while true; do
        # Check job status
        local job_status=$(kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "")
        local job_failed=$(kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
        
        if [[ "$job_status" == "True" ]]; then
            log "Job $job_name completed successfully"
            return 0
        elif [[ "$job_failed" == "True" ]]; then
            error "Job $job_name failed"
            kubectl describe job "$job_name" -n "$NAMESPACE"
            kubectl logs -l job-name="$job_name" -n "$NAMESPACE" --tail=20
            return 1
        fi
        
        # Check timeout
        local now=$(date +%s)
        if (( now - start_ts >= TIMEOUT_SECONDS )); then
            error "Job $job_name timed out after ${TIMEOUT_SECONDS}s"
            kubectl describe job "$job_name" -n "$NAMESPACE"
            return 1
        fi
        
        # Show progress from ClickHouse if available
        if [[ -n "$CH_POD" ]]; then
            local episodes=$(get_current_episodes "$exp_name" 2>/dev/null || echo "0")
            log "Job: $job_name, Episodes: $episodes/$TARGET_EPISODES"
        fi
        
        sleep "$POLL_SECONDS"
    done
}

# Get current episode count from ClickHouse
get_current_episodes() {
    local exp_name=$1
    
    local sql="SELECT max(toInt64(JSONExtractString(message, 'episode'))) AS max_episode 
               FROM system.text_log 
               WHERE message LIKE '%experiment_name%${exp_name}%' 
               AND message LIKE '%episode%'"
    
    local result=$(kubectl exec -n "$CH_NAMESPACE" "$CH_POD" -- clickhouse-client \
        -u "$CH_USER" --password "$CH_PASS" \
        --format=TSVRaw \
        -q "$sql" 2>/dev/null | tr -d '\n' | grep -o '[0-9]*' || echo "0")
    
    echo "${result:-0}"
}

# Setup persistent environment service
setup_environment() {
    log "Setting up persistent environment service..."
    
    # Check if env service already exists
    if helm status "$ENV_RELEASE" -n "$NAMESPACE" >/dev/null 2>&1; then
        log "Environment service '$ENV_RELEASE' already exists, reusing it"
        return 0
    fi
    
    # Create environment-only values
    cat > "env-only.yaml" <<EOF
env:
  enabled: true
  args:
    fps: 100
    seed: 1
    maxStepsWithoutFood: 1000
    maxSnakes: 10  # Allow multiple agents

inference:
  enabled: false

experiment:
  name: "persistent-env"
EOF
    
    log "Installing persistent environment service..."
    helm install "$ENV_RELEASE" "$CHART_DIR" \
        --namespace "$NAMESPACE" \
        -f "$BASE_VALUES" -f "env-only.yaml" \
        --wait --timeout=5m
    
    # Verify environment is healthy
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=snake-rl-env -n "$NAMESPACE" --timeout=300s
    log "Environment service is ready"
    
    rm -f "env-only.yaml"
}

# Run single experiment
run_experiment() {
    local grid=$1
    local beta=$2
    local lr=$3
    local counter=$4
    
    # Generate unique identifiers
    local beta_safe=$(echo "$beta" | sed 's/\.//g')
    local lr_safe=$(echo "$lr" | sed 's/\.//g')
    local exp_name="grid_${grid}_beta_${beta_safe}_lr_${lr_safe}"
    local release_name="exp-${counter}-$(date +%s)"
    local snake_id="agent_${counter}_$(date +%s)"
    
    log "Starting experiment: $exp_name"
    log "Release: $release_name, Agent: $snake_id"
    
    # Create experiment-specific configuration
    local config_file="override-${counter}.yaml"
    cat > "$config_file" <<EOF
env:
  enabled: false  # Use existing persistent env

inference:
  enabled: true
  runAsJob: true
  args:
    snakeId: "${snake_id}"
    envHost: "${ENV_RELEASE}-env:5000"  # Connect to persistent env
    maxEpisodes: ${TARGET_EPISODES}
    learningRate: ${lr}
    beta: ${beta}
    batchSize: 10
    gamma: 0.99

experiment:
  name: "${exp_name}"
EOF
    
    # Deploy inference job
    log "Deploying inference job..."
    if ! helm install "$release_name" "$CHART_DIR" \
        --namespace "$NAMESPACE" \
        -f "$BASE_VALUES" -f "$config_file" \
        --wait --timeout=2m; then
        error "Failed to install experiment $exp_name"
        rm -f "$config_file"
        return 1
    fi
    
    # Wait for job completion
    local job_name="${release_name}-rl-job"
    if wait_for_job_completion "$job_name" "$exp_name"; then
        log "Experiment $exp_name completed successfully"
        local final_episodes=$(get_current_episodes "$exp_name")
        log "Final episodes: $final_episodes"
    else
        error "Experiment $exp_name failed"
        # Keep logs for debugging
        kubectl logs -l job-name="$job_name" -n "$NAMESPACE" > "failed-${exp_name}.log" || true
    fi
    
    # Cleanup experiment
    log "Cleaning up experiment $exp_name"
    helm uninstall "$release_name" --namespace "$NAMESPACE" || warn "Failed to uninstall $release_name"
    rm -f "$config_file"
    
    log "Experiment $exp_name finished\n"
}

# Main execution
main() {
    log "Starting reliable experiment runner..."
    
    # Setup
    kubectl create namespace "$NAMESPACE" 2>/dev/null || log "Namespace $NAMESPACE already exists"
    setup_clickhouse || warn "ClickHouse setup failed, monitoring will be limited"
    setup_environment
    
    # Mark for environment cleanup on exit
    export CLEANUP_ENV=false  # Set to true if you want to cleanup env on exit
    
    # Run experiments
    local counter=1
    local total_experiments=$((${#grid_sizes[@]} * ${#beta_values[@]} * ${#learning_rates[@]}))
    log "Running $total_experiments experiments..."
    
    for grid in "${grid_sizes[@]}"; do
        for beta in "${beta_values[@]}"; do
            for lr in "${learning_rates[@]}"; do
                log "Experiment $counter/$total_experiments"
                run_experiment "$grid" "$beta" "$lr" "$counter"
                ((counter++))
                
                # Brief pause between experiments
                sleep 5
            done
        done
    done
    
    log "All experiments completed!"
}

# Argument parsing
case "${1:-run}" in
    "setup-env")
        kubectl create namespace "$NAMESPACE" 2>/dev/null || true
        setup_environment
        log "Persistent environment ready. Run experiments with: $0 run"
        export CLEANUP_ENV=false
        ;;
    "cleanup-env")
        helm uninstall "$ENV_RELEASE" --namespace "$NAMESPACE" || true
        log "Environment cleaned up"
        ;;
    "run")
        main
        ;;
    *)
        echo "Usage: $0 {setup-env|cleanup-env|run}"
        echo "  setup-env   - Setup persistent environment only"
        echo "  cleanup-env - Remove persistent environment"
        echo "  run         - Run all experiments (default)"
        exit 1
        ;;
esac