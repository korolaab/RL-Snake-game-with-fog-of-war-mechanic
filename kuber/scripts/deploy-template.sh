#!/bin/bash
# Deployment Script with Multiple Repository Support
# Supports Bitnami, OCI, and direct chart installation

set -e

# Configuration
NAMESPACE="data-stack"
ENVIRONMENT="${ENVIRONMENT:-dev}"
CLICKHOUSE_RELEASE="clickhouse"
RABBITMQ_RELEASE="rabbitmq"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}🚀 Deploying Data Stack - Environment: ${ENVIRONMENT}${NC}"

# Check if credentials are set
check_credentials() {
    if [ ! -f "k8s/data-stack/clickhouse/values.yaml" ]; then
        echo -e "${RED}❌ ClickHouse values.yaml not found!${NC}"
        echo "Copy from values-template.yaml and set credentials"
        exit 1
    fi
    
    if [ ! -f "k8s/data-stack/rabbitmq/values.yaml" ]; then
        echo -e "${RED}❌ RabbitMQ values.yaml not found!${NC}"
        echo "Copy from values-template.yaml and set credentials"
        exit 1
    fi
}

# Add Helm repositories with fallback options
add_helm_repos() {
    echo -e "${YELLOW}📦 Setting up Helm repositories...${NC}"
    
    # Try Bitnami first
    if helm repo add bitnami https://charts.bitnami.com/bitnami 2>/dev/null; then
        echo -e "${GREEN}✅ Bitnami repository added successfully${NC}"
        REPO_PREFIX="bitnami"
    else
        echo -e "${YELLOW}⚠️  Bitnami repository failed, trying alternatives...${NC}"
        
        # Try OCI registry
        if helm registry login registry-1.docker.io -u anonymous 2>/dev/null; then
            echo -e "${GREEN}✅ Using Bitnami OCI registry${NC}"
            REPO_PREFIX="oci://registry-1.docker.io/bitnamicharts"
        else
            echo -e "${YELLOW}⚠️  OCI failed, downloading charts directly...${NC}"
            REPO_PREFIX="direct"
        fi
    fi
    
    # Update repositories if using traditional repos
    if [ "$REPO_PREFIX" = "bitnami" ]; then
        helm repo update
    fi
}

# Download charts directly if repositories fail
download_chart_direct() {
    local chart_name=$1
    local chart_version=${2:-"latest"}
    
    echo -e "${YELLOW}📥 Downloading $chart_name chart directly...${NC}"
    
    # Create charts directory
    mkdir -p charts
    
    case $chart_name in
        "clickhouse")
            # Download ClickHouse chart from GitHub releases or Artifact Hub
            CHART_URL="https://github.com/bitnami/charts/releases/download/clickhouse-6.2.17/clickhouse-6.2.17.tgz"
            ;;
        "rabbitmq")
            # Download RabbitMQ chart from GitHub releases
            CHART_URL="https://github.com/bitnami/charts/releases/download/rabbitmq-14.6.6/rabbitmq-14.6.6.tgz"
            ;;
    esac
    
    if curl -L -o "charts/${chart_name}.tgz" "$CHART_URL" 2>/dev/null; then
        echo -e "${GREEN}✅ Downloaded $chart_name chart${NC}"
        echo "charts/${chart_name}.tgz"
    else
        echo -e "${RED}❌ Failed to download $chart_name chart${NC}"
        return 1
    fi
}

# Deploy with different methods based on available repository
deploy_clickhouse() {
    echo -e "${YELLOW}🗃️  Deploying ClickHouse...${NC}"
    
    case $REPO_PREFIX in
        "bitnami")
            helm upgrade --install $CLICKHOUSE_RELEASE bitnami/clickhouse \
                --namespace $NAMESPACE \
                --values k8s/data-stack/clickhouse/values.yaml \
                --values helm/values/${ENVIRONMENT}/clickhouse-values.yaml \
                --wait --timeout=15m
            ;;
        "oci://registry-1.docker.io/bitnamicharts")
            helm upgrade --install $CLICKHOUSE_RELEASE oci://registry-1.docker.io/bitnamicharts/clickhouse \
                --namespace $NAMESPACE \
                --values k8s/data-stack/clickhouse/values.yaml \
                --values helm/values/${ENVIRONMENT}/clickhouse-values.yaml \
                --wait --timeout=15m
            ;;
        "direct")
            CHART_PATH=$(download_chart_direct "clickhouse")
            helm upgrade --install $CLICKHOUSE_RELEASE $CHART_PATH \
                --namespace $NAMESPACE \
                --values k8s/data-stack/clickhouse/values.yaml \
                --values helm/values/${ENVIRONMENT}/clickhouse-values.yaml \
                --wait --timeout=15m
            ;;
    esac
}

deploy_rabbitmq() {
    echo -e "${YELLOW}🐰 Deploying RabbitMQ...${NC}"
    
    case $REPO_PREFIX in
        "bitnami")
            helm upgrade --install $RABBITMQ_RELEASE bitnami/rabbitmq \
                --namespace $NAMESPACE \
                --values k8s/data-stack/rabbitmq/values.yaml \
                --values helm/values/${ENVIRONMENT}/rabbitmq-values.yaml \
                --wait --timeout=10m
            ;;
        "oci://registry-1.docker.io/bitnamicharts")
            helm upgrade --install $RABBITMQ_RELEASE oci://registry-1.docker.io/bitnamicharts/rabbitmq \
                --namespace $NAMESPACE \
                --values k8s/data-stack/rabbitmq/values.yaml \
                --values helm/values/${ENVIRONMENT}/rabbitmq-values.yaml \
                --wait --timeout=10m
            ;;
        "direct")
            CHART_PATH=$(download_chart_direct "rabbitmq")
            helm upgrade --install $RABBITMQ_RELEASE $CHART_PATH \
                --namespace $NAMESPACE \
                --values k8s/data-stack/rabbitmq/values.yaml \
                --values helm/values/${ENVIRONMENT}/rabbitmq-values.yaml \
                --wait --timeout=10m
            ;;
    esac
}

# Main deployment function
deploy() {
    check_credentials
    add_helm_repos
    
    # Apply base configurations
    echo -e "${YELLOW}📁 Creating namespaces...${NC}"
    kubectl apply -f k8s/namespaces/ 2>/dev/null || true
    
    echo -e "${YELLOW}💾 Applying storage classes...${NC}"
    kubectl apply -f k8s/storage/ 2>/dev/null || true
    
    # Deploy services
    deploy_clickhouse
    deploy_rabbitmq
    
    # Apply network policies
    echo -e "${YELLOW}🔒 Applying network policies...${NC}"
    kubectl apply -f k8s/data-stack/network-policies.yaml 2>/dev/null || true
    
    echo -e "${GREEN}✅ Deployment completed!${NC}"
    echo ""
    echo -e "${BLUE}📊 Checking status...${NC}"
    kubectl get pods -n $NAMESPACE
    echo ""
    echo -e "${GREEN}🔗 Connection information:${NC}"
    echo "  Run: make urls       # Get connection URLs"
    echo "  Run: make passwords  # View passwords"
    echo "  Run: make port-forward # Access services locally"
}

# Run deployment
deploy "$@"
