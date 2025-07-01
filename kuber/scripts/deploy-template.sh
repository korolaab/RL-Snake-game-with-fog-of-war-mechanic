#!/bin/bash
# Deployment Script Template
# Copy to deploy.sh and customize

set -e

# Configuration
NAMESPACE="data-stack"
ENVIRONMENT="${ENVIRONMENT:-dev}"  # dev, staging, production

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Deploying Data Stack - Environment: ${ENVIRONMENT}${NC}"

# Check if credentials are set
check_credentials() {
    if [ ! -f "k8s/data-stack/clickhouse/values.yaml" ]; then
        echo "❌ ClickHouse values.yaml not found!"
        echo "Copy from values-template.yaml and set credentials"
        exit 1
    fi
    
    if [ ! -f "k8s/data-stack/rabbitmq/values.yaml" ]; then
        echo "❌ RabbitMQ values.yaml not found!"
        echo "Copy from values-template.yaml and set credentials"
        exit 1
    fi
}

# Add Helm repositories
add_helm_repos() {
    echo -e "${YELLOW}📦 Adding Helm repositories...${NC}"
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
}

# Deploy function
deploy() {
    check_credentials
    add_helm_repos
    
    # Apply base configurations
    kubectl apply -f k8s/namespaces/
    kubectl apply -f k8s/storage/
    
    # Deploy ClickHouse
    helm upgrade --install clickhouse bitnami/clickhouse \
        --namespace ${NAMESPACE} \
        --values k8s/data-stack/clickhouse/values.yaml \
        --values helm/values/${ENVIRONMENT}/clickhouse-values.yaml \
        --wait --timeout=15m
    
    # Deploy RabbitMQ
    helm upgrade --install rabbitmq bitnami/rabbitmq \
        --namespace ${NAMESPACE} \
        --values k8s/data-stack/rabbitmq/values.yaml \
        --values helm/values/${ENVIRONMENT}/rabbitmq-values.yaml \
        --wait --timeout=10m
    
    # Apply network policies
    kubectl apply -f k8s/data-stack/network-policies.yaml
    
    echo -e "${GREEN}✅ Deployment completed!${NC}"
}

deploy "$@"
