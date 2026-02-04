#!/bin/bash

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "=== Snake RL Minikube Setup ==="
echo ""

# Check prerequisites
print_step "Checking prerequisites..."

if ! command -v minikube &> /dev/null; then
    print_error "minikube not found. Please install: https://minikube.sigs.k8s.io/docs/start/"
    exit 1
fi

if ! command -v helm &> /dev/null; then
    print_error "helm not found. Please install: https://helm.sh/docs/intro/install/"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    print_error "kubectl not found. Please install: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

print_info "All prerequisites found!"
echo ""

# Start minikube
print_step "Starting minikube..."

if minikube status &> /dev/null; then
    print_info "Minikube already running"
else
    print_info "Starting minikube with 4 CPUs and 8GB memory..."
    minikube start --cpus=4 --memory=8192
fi

# Verify minikube is running
if ! minikube status | grep -q "Running"; then
    print_error "Failed to start minikube"
    exit 1
fi

print_info "Minikube is running"
echo ""

# Build images in minikube's Docker
print_step "Building Docker images in minikube..."

print_info "Configuring Docker environment..."
eval $(minikube docker-env)

print_info "Building base image (this will take 5-10 minutes)..."
cd ../../services
docker build -t korolaab/snake_rl_base:latest -f Dockerfile . || {
    print_error "Failed to build base image"
    exit 1
}

print_info "Building Clock service..."
docker build -t localhost:5000/snake-rl/clock:latest -f Clock/Dockerfile Clock/ || {
    print_error "Failed to build Clock service"
    exit 1
}

print_info "Building Environment service..."
docker build -t localhost:5000/snake-rl/env:latest -f Env/Dockerfile Env/ || {
    print_error "Failed to build Environment service"
    exit 1
}

print_info "Building Inference service..."
docker build -t localhost:5000/snake-rl/inference:latest -f Inference/Dockerfile Inference/ || {
    print_error "Failed to build Inference service"
    exit 1
}

print_info "All images built successfully!"
echo ""

# Install Helm chart
print_step "Installing Helm chart..."

cd ../k8s/snake-rl

print_info "Linting Helm chart..."
helm lint . || {
    print_error "Helm chart has errors"
    exit 1
}

print_info "Installing snake-rl chart with minikube values..."
helm install snake-rl . -f values-minikube.yaml || {
    print_error "Failed to install Helm chart"
    exit 1
}

echo ""
print_step "Setup complete!"
echo ""
print_info "Useful commands:"
echo "  Watch pods:      kubectl get pods -w"
echo "  View logs:       kubectl logs job/snake-rl -c clock -f"
echo "  Check status:    kubectl get jobs"
echo "  Delete job:      helm uninstall snake-rl"
echo ""
print_info "The experiment will run for 3000 episodes (~10-30 minutes)"
print_info "Logs and models are saved in the PVC: snake-rl-logs"
