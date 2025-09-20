#!/bin/bash
set -euo pipefail

# Find optimal game over conditions experiment runner
# Testing max_steps_without_food parameter with different beta values
BASE_VALUES="k8s/snake-rl/values.yaml"
CHART_DIR="k8s/snake-rl"
NAMESPACE="experiments"

# Experiment configuration - Game Over Conditions Study
# max_steps_without_food: 30 -> 1000 with 50% increments
# Starting at 30, each step = previous * 1.5, final step set to 1000
max_steps_values=(30 45 67 100 150 225 337 505 757 1000)
beta_values=(0 0.0001 0.16)
learning_rates=(0.001)  # Fixed learning rate
TARGET_EPISODES=1000    # Long runs to find optimal conditions
JOB_TIMEOUT=7200       # 2 hours per experiment (1000 episodes)

# Fixed environment parameters for consistency
GRID_SIZE=11
VISION_RADIUS=5
FPS=100

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
    log "This may take up to 2 hours for 1000 episodes..."
    
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
        kubectl logs -l job-name="$job_name" -n "$NAMESPACE" --tail=20 2>/dev/null || error "Could not retrieve logs"
        
        return 1
    fi
}

# Run single experiment with dedicated ENV + JOB
run_experiment() {
    local max_steps=$1
    local beta=$2
    local lr=$3
    local counter=$4
    
    # Generate unique identifiers
    local beta_safe=$(echo "$beta" | sed 's/\./_/g')
    local lr_safe=$(echo "$lr" | sed 's/\./_/g')
    local experiment_name="game_over_maxsteps${max_steps}_beta${beta_safe}_lr${lr_safe}"
    # Helm release name must be lowercase, alphanumeric+hyphens only, max 53 chars
    local release_name=$(echo "go-ms${max_steps}-b${beta_safe}-${counter}" | sed 's/_/-/g' | tr '[:upper:]' '[:lower:]')
    local snake_id="adam"
    
    log "=== Game Over Experiment $counter ==="
    log "Experiment: $experiment_name"
    log "Max steps without food: $max_steps"
    log "Beta (entropy): $beta"
    log "Episodes: $TARGET_EPISODES"
    log "Release: $release_name, Agent: $snake_id"
    
    # Create experiment configuration (ENV + JOB)
    local config_file="config-${release_name}.yaml"
    cat > "$config_file" <<EOF
env:
  enabled: true
  args:
    gridWidth: ${GRID_SIZE}
    gridHeight: ${GRID_SIZE}
    fps: ${FPS}
    seed: ${counter}
    maxStepsWithoutFood: ${max_steps}
    visionRadius: ${VISION_RADIUS}
    rewardConfig: '{"alive": 0.1, "eat_food": 1.0, "game_over": -1.0}'

inference:
  enabled: true
  runAsJob: true
  args:
    snakeId: "${snake_id}"
    envHost: "${release_name}-env:5000"
    maxEpisodes: ${TARGET_EPISODES}
    learningRate: ${lr}
    beta: ${beta}
    batchSize: 1
    gamma: 0.95

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
    # Job name follows Helm template: ${release_name}-snake-rl-rl-job
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
    log "Starting Game Over Conditions Optimization Study..."
    log "=========================================="
    log "Goal: Find optimal max_steps_without_food parameter"
    log "Testing values: ${max_steps_values[*]}"
    log "Beta values: ${beta_values[*]}"
    log "Episodes per experiment: $TARGET_EPISODES"
    log "Job timeout: ${JOB_TIMEOUT}s (2 hours) per experiment"
    log "=========================================="
    
    # Setup namespace
    kubectl create namespace "$NAMESPACE" 2>/dev/null || log "Namespace $NAMESPACE already exists"
    
    # Calculate total experiments
    local total_experiments=$((${#max_steps_values[@]} * ${#beta_values[@]} * ${#learning_rates[@]}))
    local counter=1
    local successful=0
    local failed=0
    
    log "Running $total_experiments experiments..."
    log "Estimated total time: ~$((total_experiments * 2)) hours"
    
    # Run all experiments
    for max_steps in "${max_steps_values[@]}"; do
        log "Debug: Starting max_steps_without_food = $max_steps"
        for beta in "${beta_values[@]}"; do
            for lr in "${learning_rates[@]}"; do
                log "\n=== Experiment $counter/$total_experiments ==="
                log "Parameters: max_steps=$max_steps, beta=$beta, lr=$lr"
                
                if run_experiment "$max_steps" "$beta" "$lr" "$counter"; then
                    successful=$((successful + 1))
                    log "✅ Success: $successful/$counter experiments"
                else
                    failed=$((failed + 1))
                    error "❌ Failed: $failed/$counter experiments"
                fi
                
                counter=$((counter + 1))
                
                # Brief pause between experiments
                if [[ $counter -le $total_experiments ]]; then
                    log "Pausing 30s before next experiment..."
                    sleep 30
                else
                    log "All experiments completed, exiting loop"
                fi
            done
        done
    done
    
    # Final summary
    log "\n=== GAME OVER CONDITIONS STUDY SUMMARY ==="
    log "Total experiments: $total_experiments"
    log "Successful: $successful"
    log "Failed: $failed"
    log "Parameters tested:"
    log "  - max_steps_without_food: ${max_steps_values[*]}"
    log "  - beta values: ${beta_values[*]}"
    log "  - episodes per experiment: $TARGET_EPISODES"
    
    if [[ $failed -eq 0 ]]; then
        log "🎉 All game over condition experiments completed successfully!"
        log "Check your data analytics to find optimal max_steps_without_food value"
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