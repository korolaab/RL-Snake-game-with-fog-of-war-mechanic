#!/bin/bash

# Deploy Snake RL to remote Kubernetes cluster
# Usage:
#   export REMOTE_REGISTRY_HOST=192.168.88.253
#   export REMOTE_USER=your_username  # optional, defaults to $USER
#   ./scripts/deploy-to-remote.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_step() { echo -e "${GREEN}[STEP]${NC} $1"; }
print_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check required environment variables
if [ -z "$REMOTE_REGISTRY_HOST" ]; then
    print_error "REMOTE_REGISTRY_HOST is not set!"
    print_info "Usage:"
    echo "  export REMOTE_REGISTRY_HOST=192.168.88.253"
    echo "  export REMOTE_USER=username  # optional"
    echo "  ./scripts/deploy-to-remote.sh"
    exit 1
fi

REMOTE_HOST="${REMOTE_REGISTRY_HOST}"
REMOTE_USER="${REMOTE_USER:-$USER}"
REGISTRY="${REMOTE_HOST}:5000"

echo "=== Snake RL Remote Deployment ==="
echo "Remote host: ${REMOTE_HOST}"
echo "Registry: ${REGISTRY}"
echo "User: ${REMOTE_USER}"
echo ""

# Generate values-remote.yaml from template
print_step "Generating values-remote.yaml from template..."
envsubst < k8s/snake-rl/values-remote.yaml.template > k8s/snake-rl/values-remote.yaml
print_info "Generated k8s/snake-rl/values-remote.yaml"
echo ""

# Copy project to remote
print_step "Copying project to remote host..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ~/snake_rl"
rsync -avz --exclude '.git' --exclude '*.pyc' --exclude '__pycache__' \
    --exclude 'logs-storage' --exclude '.venv' --exclude 'venv' \
    ./ ${REMOTE_USER}@${REMOTE_HOST}:~/snake_rl/

print_info "Project copied"
echo ""

# Build and push images on remote
print_step "Building Docker images on remote host..."
ssh ${REMOTE_USER}@${REMOTE_HOST} << ENDSSH
cd ~/snake_rl/services

echo "[Remote] Building base image..."
docker build -t ${REGISTRY}/snake-rl/base:latest -f Dockerfile .

echo "[Remote] Building Clock service..."
docker build -t ${REGISTRY}/snake-rl/clock:latest -f Clock/Dockerfile Clock/

echo "[Remote] Building Environment service..."
docker build -t ${REGISTRY}/snake-rl/env:latest -f Env/Dockerfile Env/

echo "[Remote] Building Inference service..."
docker build -t ${REGISTRY}/snake-rl/inference:latest -f Inference/Dockerfile Inference/

echo "[Remote] Pushing images to registry..."
docker push ${REGISTRY}/snake-rl/clock:latest
docker push ${REGISTRY}/snake-rl/env:latest
docker push ${REGISTRY}/snake-rl/inference:latest

echo "[Remote] Images built and pushed successfully!"
ENDSSH

print_info "Images ready in registry"
echo ""

# Deploy via Helm
print_step "Deploying with Helm..."

# Check if deployment already exists
if kubectl get job snake-rl &>/dev/null; then
    print_info "Existing deployment found. Deleting..."
    helm uninstall snake-rl || true
    kubectl delete job snake-rl --ignore-not-found=true
    sleep 5
fi

print_info "Installing Helm chart..."
helm install snake-rl k8s/snake-rl/ -f k8s/snake-rl/values-remote.yaml

print_info "Deployment initiated!"
echo ""

# Show status
print_step "Deployment complete!"
echo ""
print_info "Monitor the experiment:"
echo "  kubectl get pods -w"
echo "  kubectl logs job/snake-rl -c clock -f"
echo ""
print_info "Check job status:"
echo "  kubectl get jobs"
echo "  kubectl describe job snake-rl"
echo ""
print_info "View results (after completion):"
echo "  kubectl logs job/snake-rl -c clock"
echo "  kubectl exec -it job/snake-rl -c clock -- ls -la /logs"
echo ""
print_info "Cleanup:"
echo "  helm uninstall snake-rl"
echo "  rm k8s/snake-rl/values-remote.yaml  # Remove generated file"
