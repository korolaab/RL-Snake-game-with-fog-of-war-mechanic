#!/bin/bash
set -euo pipefail

# Reliable experiment runner using ONLY Kubernetes Job status
BASE_VALUES="k8s/snake-rl/values.yaml"
CHART_DIR="k8s/snake-rl"
NAMESPACE="experiments"

# Experiment configuration - 5 test experiments
grid_sizes=(7 9 11 13 15)
beta_values=(0.001)
learning_rates=(0.001)
TARGET_EPISODES=10
JOB_TIMEOUT=600  # 10 minutes per job

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"; }

# Cleanup function
cleanup_experiment() {
    local release_name=$1
    log "Cleaning up experiment: $release_name"
    set +e  # Temporarily disable exit on error for cleanup
    helm uninstall "$release_name" --namespace "$NAMESPACE" 2>/dev/null
    rm -f "config-${release_name}.yaml" 2>/dev/null
    set -e  # Re-enable exit on error
    log "Cleanup completed for: $release_name"
}

# Wait for job completion using ONLY Kubernetes job status
wait_for_job_completion() {
    local job_name=$1
    local experiment_name=$2
    
    log "Waiting for job: $job_name (experiment: $experiment_name)"
    
    # Use kubectl wait - most reliable method
    if kubectl wait --for=condition=complete job/"$job_name" \
        --namespace="$NAMESPACE" \
        --timeout="${JOB_TIMEOUT}s" 2>/dev/null; then
        log "✅ Job completed successfully: $experiment_name"
        return 0
    else
        # Check if job failed or timed out
        local failed=$(kubectl get job "$job_name" -n "$NAMESPACE" \
            -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
        
        if [[ "$failed" == "True" ]]; then
            error "❌ Job failed: $experiment_name"
            # Get failure reason
            local reason=$(kubectl get job "$job_name" -n "$NAMESPACE" \
                -o jsonpath='{.status.conditions[?(@.type=="Failed")].message}' 2>/dev/null || echo "Unknown")
            error "Failure reason: $reason"
        else
            error "⏰ Job timed out after ${JOB_TIMEOUT}s: $experiment_name"
        fi
        
        # Show recent logs for debugging
        log "Recent logs:"
        kubectl logs -l job-name="$job_name" -n "$NAMESPACE" --tail=10 2>/dev/null || error "Could not retrieve logs"
        
        return 1
    fi
}

# Run single experiment with dedicated ENV + JOB
run_experiment() {
    local grid=$1
    local beta=$2
    local lr=$3
    local counter=$4
    
    # Generate unique identifiers
    local beta_safe=$(echo "$beta" | sed 's/\./_/g')
    local lr_safe=$(echo "$lr" | sed 's/\./_/g')
    local experiment_name="test_script_grid${grid}_beta${beta_safe}_lr${lr_safe}"
    local release_name="exp-${counter}-$(date +%s | tail -c 6)"
    local snake_id="agent_${counter}"
    
    log "=== Experiment $counter: $experiment_name ==="
    log "Release: $release_name, Agent: $snake_id"
    
    # Create experiment configuration (ENV + JOB)
    local config_file="config-${release_name}.yaml"
    cat > "$config_file" <<EOF
env:
  enabled: true
  args:
    gridWidth: ${grid}
    gridHeight: ${grid}
    fps: 100
    seed: ${counter}
    maxStepsWithoutFood: 1000
    visionRadius: 5

inference:
  enabled: true
  runAsJob: true
  args:
    snakeId: "${snake_id}"
    envHost: "${release_name}-env:5000"
    maxEpisodes: ${TARGET_EPISODES}
    learningRate: ${lr}
    beta: ${beta}
    batchSize: 10
    gamma: 0.99

experiment:
  name: "${experiment_name}"
EOF
    
    # Deploy experiment (both ENV and JOB)
    log "Deploying experiment..."
    if ! helm install "$release_name" "$CHART_DIR" \
        --namespace "$NAMESPACE" \
        -f "$BASE_VALUES" -f "$config_file"; then
        error "Failed to deploy experiment: $experiment_name"
        cleanup_experiment "$release_name"
        return 1
    fi
    
    # Wait for job completion  
    # Job name follows Helm template: {{ include "snake-rl.fullname" . }}-rl-job
    # Which becomes: ${release_name}-snake-rl-rl-job
    local job_name="${release_name}-snake-rl-rl-job"
    local success=false
    
    if wait_for_job_completion "$job_name" "$experiment_name"; then
        log "🎉 Experiment completed: $experiment_name"
        success=true
    else
        error "💥 Experiment failed: $experiment_name"
        success=false
    fi
    
    # Always cleanup
    cleanup_experiment "$release_name"
    
    log "Debug: Cleanup completed, about to return from run_experiment"
    if [[ "$success" == "true" ]]; then
        log "Debug: Returning 0 (success) from run_experiment"
        return 0
    else
        log "Debug: Returning 1 (failure) from run_experiment"
        return 1
    fi
}

# Main execution
main() {
    log "Starting job-status-only experiment runner..."
    
    # Setup namespace
    kubectl create namespace "$NAMESPACE" 2>/dev/null || log "Namespace $NAMESPACE already exists"
    
    # Calculate total experiments
    local total_experiments=$((${#grid_sizes[@]} * ${#beta_values[@]} * ${#learning_rates[@]}))
    local counter=1
    local successful=0
    local failed=0
    
    log "Running $total_experiments experiments with $TARGET_EPISODES episodes each..."
    log "Job timeout: ${JOB_TIMEOUT}s per experiment"
    log "Grid sizes: ${grid_sizes[*]}"
    log "Beta values: ${beta_values[*]}"
    log "Learning rates: ${learning_rates[*]}"
    
    # Run all experiments
    for grid in "${grid_sizes[@]}"; do
        log "Debug: Starting grid size $grid"
        for beta in "${beta_values[@]}"; do
            for lr in "${learning_rates[@]}"; do
                log "\n--- Experiment $counter/$total_experiments ---"
                
                log "Debug: About to run experiment $counter"
                if run_experiment "$grid" "$beta" "$lr" "$counter"; then
                    log "Debug: run_experiment returned 0, about to increment successful"
                    successful=$((successful + 1))
                    log "Debug: successful incremented to $successful"
                    log "✅ Success: $successful/$counter experiments"
                    log "Debug: run_experiment returned success"
                else
                    log "Debug: run_experiment returned non-zero, about to increment failed"
                    failed=$((failed + 1))
                    log "Debug: failed incremented to $failed"
                    error "❌ Failed: $failed/$counter experiments"
                    log "Debug: run_experiment returned failure"
                fi
                
                log "Debug: About to increment counter from $counter"
                ((counter++))
                log "Debug: Counter incremented to $counter, continuing to next..."
                
                # Brief pause between experiments
                if [[ $counter -le $total_experiments ]]; then
                    log "Pausing 10s before next experiment..."
                    sleep 10
                else
                    log "All experiments completed, exiting loop"
                fi
            done
        done
    done
    
    # Final summary
    log "\n=== EXPERIMENT SUMMARY ==="
    log "Total experiments: $total_experiments"
    log "Successful: $successful"
    log "Failed: $failed"
    
    if [[ $failed -eq 0 ]]; then
        log "🎉 All experiments completed successfully!"
        return 0
    else
        error "💥 $failed experiments failed"
        return 1
    fi
}

# Cleanup on interrupt only (not normal exit)
trap 'log "Script interrupted, cleaning up..."; kubectl delete jobs -l app.kubernetes.io/instance --all -n "$NAMESPACE" 2>/dev/null || true; exit 1' INT TERM

# Run main
main "$@"