#!/bin/bash
set -e

REGISTRY="localhost:5000"
VERSION="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_registry() {
    echo_info "Checking if local Docker registry is running..."
    if ! curl -s http://localhost:5000/v2/ > /dev/null; then
        echo_error "Local Docker registry is not running on localhost:5000"
        echo_info "Start it with: ./scripts/start-registry.sh"
        exit 1
    fi
    echo_info "Local Docker registry is running"
}

build_base_image() {
    echo_info "Building base image korolaab/snake_rl_base..."
    docker build -t korolaab/snake_rl_base:${VERSION} ./services/
    docker tag korolaab/snake_rl_base:${VERSION} ${REGISTRY}/korolaab/snake_rl_base:${VERSION}
    docker push ${REGISTRY}/korolaab/snake_rl_base:${VERSION}
    echo_info "Base image built and pushed successfully"
}

build_service_images() {
    local services=("Env"  "Inference" ) #"Training"
    
    for service in "${services[@]}"; do
        echo_info "Building ${service} service..."
        
        service_lower=$(echo "$service" | tr '[:upper:]' '[:lower:]')
        image_name="snake-rl/${service_lower}"
        
        # Build the image
        docker build -f ./services/${service}/Dockerfile \
                    -t ${image_name}:${VERSION} \
                    ./services/${service}
        
        # Tag for registry
        docker tag ${image_name}:${VERSION} ${REGISTRY}/${image_name}:${VERSION}
        
        # Push to registry
        docker push ${REGISTRY}/${image_name}:${VERSION}
        
        echo_info "${service} service built and pushed successfully"
    done
}

main() {
    echo_info "Starting build and push process..."
    
    check_registry
    build_base_image
    build_service_images

    echo_info "All images built and pushed successfully!"
    echo_info "Images available in registry:"
    echo "  - ${REGISTRY}/korolaab/snake_rl_base:${VERSION}"
    echo "  - ${REGISTRY}/snake-rl/env:${VERSION}"
   # echo "  - ${REGISTRY}/snake-rl/training:${VERSION}"
    echo "  - ${REGISTRY}/snake-rl/inference:${VERSION}"
}

main "$@"