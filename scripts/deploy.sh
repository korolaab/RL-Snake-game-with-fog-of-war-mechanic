#!/bin/bash
# Intelligent Deployment Script with Proper Secret Management
# Automatically detects chart sources and ensures secrets exist

set -e

# Configuration
NAMESPACE="data-stack"
ENVIRONMENT="${ENVIRONMENT:-dev}"
CLICKHOUSE_RELEASE="clickhouse"
RABBITMQ_RELEASE="rabbitmq"

# Chart versions
CLICKHOUSE_VERSION="6.2.17"
RABBITMQ_VERSION="14.6.6"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}🚀 Intelligent Data Stack Deployment${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo ""

# Global variables
DEPLOYMENT_METHOD=""

# Utility functions
test_url() {
    local url=$1
    local timeout=${2:-5}
    curl -s --max-time $timeout --head "$url" >/dev/null 2>&1
}

# Check and create secrets if needed
ensure_secrets() {
    echo -e "${YELLOW}🔐 Ensuring secrets exist...${NC}"
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
    
    # Check ClickHouse secret
    if ! kubectl get secret clickhouse-credentials -n $NAMESPACE >/dev/null 2>&1; then
        echo -e "${CYAN}Creating ClickHouse secret...${NC}"
        if [ -f .secrets/clickhouse_password ]; then
            kubectl create secret generic clickhouse-credentials \
                --from-file=password=.secrets/clickhouse_password \
                -n $NAMESPACE
        else
            # Generate password if file doesn't exist
            mkdir -p .secrets
            openssl rand -base64 20 | tr -d "=+/" | cut -c1-16 > .secrets/clickhouse_password
            kubectl create secret generic clickhouse-credentials \
                --from-file=password=.secrets/clickhouse_password \
                -n $NAMESPACE
        fi
        echo -e "${GREEN}✅ ClickHouse secret created${NC}"
    else
        echo -e "${BLUE}ℹ️  ClickHouse secret already exists${NC}"
    fi
    
    # Check RabbitMQ secret
    if ! kubectl get secret rabbitmq-credentials -n $NAMESPACE >/dev/null 2>&1; then
        echo -e "${CYAN}Creating RabbitMQ secret...${NC}"
        if [ -f .secrets/rabbitmq_password ]; then
            kubectl create secret generic rabbitmq-credentials \
                --from-file=password=.secrets/rabbitmq_password \
                -n $NAMESPACE
        else
            # Generate password if file doesn't exist
            mkdir -p .secrets
            openssl rand -base64 20 | tr -d "=+/" | cut -c1-16 > .secrets/rabbitmq_password
            kubectl create secret generic rabbitmq-credentials \
                --from-file=password=.secrets/rabbitmq_password \
                -n $NAMESPACE
        fi
        echo -e "${GREEN}✅ RabbitMQ secret created${NC}"
    else
        echo -e "${BLUE}ℹ️  RabbitMQ secret already exists${NC}"
    fi
}

# Auto-detect deployment method
detect_deployment_method() {
    echo -e "${YELLOW}🔍 Auto-detecting chart sources...${NC}"
    
    # Method 1: Try traditional Helm repositories
    echo -e "${CYAN}  Testing Bitnami repository...${NC}"
    if helm repo add bitnami https://charts.bitnami.com/bitnami >/dev/null 2>&1; then
        if helm repo update >/dev/null 2>&1 && helm search repo bitnami/clickhouse >/dev/null 2>&1; then
            DEPLOYMENT_METHOD="helm_repo"
            echo -e "${GREEN}📦 Using: Bitnami Helm Repository${NC}"
            return 0
        else
            helm repo remove bitnami >/dev/null 2>&1 || true
        fi
    fi
    
    # Method 2: Try OCI registry
    echo -e "${CYAN}  Testing OCI registry...${NC}"
    if helm show chart oci://registry-1.docker.io/bitnamicharts/clickhouse --version "$CLICKHOUSE_VERSION" >/dev/null 2>&1; then
        DEPLOYMENT_METHOD="oci_registry"
        echo -e "${GREEN}📦 Using: OCI Registry (Docker Hub)${NC}"
        return 0
    fi
    
    # Method 3: Try direct downloads
    echo -e "${CYAN}  Testing direct downloads...${NC}"
    local clickhouse_url="https://github.com/bitnami/charts/releases/download/clickhouse-${CLICKHOUSE_VERSION}/clickhouse-${CLICKHOUSE_VERSION}.tgz"
    if test_url "$clickhouse_url" 3; then
        DEPLOYMENT_METHOD="direct_download"
        echo -e "${GREEN}📦 Using: Direct Chart Downloads${NC}"
        return 0
    fi
    
    # Method 4: Try repository clone
    echo -e "${CYAN}  Testing repository clone...${NC}"
    if test_url "https://github.com/bitnami/charts.git" 3; then
        DEPLOYMENT_METHOD="repo_clone"
        echo -e "${GREEN}📦 Using: Repository Clone${NC}"
        return 0
    fi
    
    echo -e "${RED}❌ No chart sources available!${NC}"
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  1. Check internet connectivity"
    echo "  2. Try again later"
    echo "  3. Check firewall settings"
    exit 1
}

# Download chart based on method
download_chart() {
    local chart_name=$1
    local chart_version=$2
    
    case $DEPLOYMENT_METHOD in
        "helm_repo")
            echo "bitnami/${chart_name}"
            ;;
        "oci_registry")
            echo "oci://registry-1.docker.io/bitnamicharts/${chart_name}"
            ;;
        "direct_download")
            local url="https://github.com/bitnami/charts/releases/download/${chart_name}-${chart_version}/${chart_name}-${chart_version}.tgz"
            local chart_file="charts/${chart_name}-${chart_version}.tgz"
            
            mkdir -p charts
            
            if [ ! -f "$chart_file" ]; then
                echo -e "${CYAN}📥 Downloading ${chart_name} chart...${NC}"
                if curl -L -o "$chart_file" "$url" --progress-bar; then
                    echo -e "${GREEN}✅ Downloaded ${chart_name}${NC}"
                else
                    echo -e "${RED}❌ Failed to download ${chart_name}${NC}"
                    exit 1
                fi
            fi
            
            echo "$chart_file"
            ;;
        "repo_clone")
            if [ ! -d "charts-repo" ]; then
                echo -e "${CYAN}📥 Cloning charts repository...${NC}"
                git clone --depth 1 https://github.com/bitnami/charts.git charts-repo >/dev/null 2>&1
            fi
            
            echo "charts-repo/bitnami/${chart_name}"
            ;;
    esac
}

# Deploy a chart
deploy_chart() {
    local release_name=$1
    local chart_name=$2
    local chart_version=$3
    local values_file=$4
    local env_values_file=$5
    local timeout=$6
    
    echo -e "${YELLOW}🚀 Deploying ${chart_name}...${NC}"
    
    local chart_path=$(download_chart "$chart_name" "$chart_version")
    
    # Build helm command
    local helm_cmd="helm upgrade --install $release_name $chart_path"
    helm_cmd="$helm_cmd --namespace $NAMESPACE"
    helm_cmd="$helm_cmd --create-namespace"
    helm_cmd="$helm_cmd --values $values_file"
    
    if [ -f "$env_values_file" ]; then
        helm_cmd="$helm_cmd --values $env_values_file"
    fi
    
    # Add version for specific methods
    if [ "$DEPLOYMENT_METHOD" = "oci_registry" ] || [ "$DEPLOYMENT_METHOD" = "helm_repo" ]; then
        helm_cmd="$helm_cmd --version $chart_version"
    fi
    
    helm_cmd="$helm_cmd --wait --timeout=${timeout}"
    
    echo -e "${CYAN}Running: $helm_cmd${NC}"
    
    if eval $helm_cmd; then
        echo -e "${GREEN}✅ ${chart_name} deployed successfully${NC}"
    else
        echo -e "${RED}❌ Failed to deploy ${chart_name}${NC}"
        
        # Show debugging info
        echo -e "${YELLOW}Debug information:${NC}"
        kubectl get pods -n $NAMESPACE | grep $release_name || true
        kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' | tail -10
        
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"
    
    # Check if config files exist
    if [ ! -f "k8s/data-stack/rabbitmq/values.yaml" ]; then
        echo -e "${RED}❌ RabbitMQ values.yaml not found!${NC}"
        echo "Run: make init"
        exit 1
    fi
    
    # Check kubectl access
    if ! kubectl cluster-info >/dev/null 2>&1; then
        echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm >/dev/null 2>&1; then
        echo -e "${RED}❌ Helm not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Clean up any failed deployments
cleanup_failed_deployments() {
    echo -e "${YELLOW}🧹 Cleaning up any failed deployments...${NC}"
    
    # Remove failed Helm releases
    helm list -n $NAMESPACE --failed -q | xargs -r helm delete -n $NAMESPACE 2>/dev/null || true
    
    # Remove failed pods
    kubectl delete pods --field-selector=status.phase=Failed -n $NAMESPACE 2>/dev/null || true
    
    # Remove pending PVCs if they exist and are stuck
    kubectl get pvc -n $NAMESPACE 2>/dev/null | grep Pending | awk '{print $1}' | xargs -r kubectl delete pvc -n $NAMESPACE 2>/dev/null || true
}

# Show post-deployment info
show_deployment_info() {
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${BLUE}Method used: $DEPLOYMENT_METHOD${NC}"
    echo ""
    
    # Show status
    echo -e "${YELLOW}📊 Current status:${NC}"
    kubectl get pods -n $NAMESPACE
    echo ""
    
    # Show secrets info
    echo -e "${YELLOW}🔑 Secrets created:${NC}"
    kubectl get secrets -n $NAMESPACE | grep -E "(clickhouse|rabbitmq)" || echo "No secrets found"
    echo ""
    
    # Show next steps
    echo -e "${GREEN}🔗 Next steps:${NC}"
    echo "  make passwords      # View passwords"
    echo "  make urls          # Get connection URLs"
    echo "  make port-forward  # Access services locally"
    echo "  make health        # Check service health"
    echo ""
    
    # Show password info
    echo -e "${CYAN}💡 Password files saved in .secrets/ directory${NC}"
    if [ -f .secrets/clickhouse_password ]; then
        echo "  ClickHouse: .secrets/clickhouse_password"
    fi
    if [ -f .secrets/rabbitmq_password ]; then
        echo "  RabbitMQ: .secrets/rabbitmq_password"
    fi
}

# Cleanup function for downloaded files
cleanup() {
    if [ "$DEPLOYMENT_METHOD" = "direct_download" ]; then
        echo -e "${YELLOW}🧹 Cleaning up downloaded charts...${NC}"
        rm -rf charts/
    elif [ "$DEPLOYMENT_METHOD" = "repo_clone" ]; then
        echo -e "${YELLOW}🧹 Cleaning up cloned repository...${NC}"
        rm -rf charts-repo/
    fi
}

# Main deployment function
main_deploy() {
    echo -e "${BLUE}Starting intelligent deployment...${NC}"
    echo ""
    
    # Pre-deployment checks
    check_prerequisites
    cleanup_failed_deployments
    
    # Ensure secrets exist
    ensure_secrets
    echo ""
    
    # Auto-detect best deployment method
    detect_deployment_method
    echo ""
    
    # Apply base configurations
    echo -e "${YELLOW}📁 Applying base configurations...${NC}"
    kubectl apply -f k8s/namespaces/ 2>/dev/null || true
    kubectl apply -f k8s/storage/ 2>/dev/null || true
    echo ""
    
    # Deploy ClickHouse
    deploy_chart "$CLICKHOUSE_RELEASE" "clickhouse" "$CLICKHOUSE_VERSION" \
        "k8s/data-stack/clickhouse/values.yaml" \
        "helm/values/${ENVIRONMENT}/clickhouse-values.yaml" \
        "15m"
    echo ""
    
    # Deploy RabbitMQ
    deploy_chart "$RABBITMQ_RELEASE" "rabbitmq" "$RABBITMQ_VERSION" \
        "k8s/data-stack/rabbitmq/values.yaml" \
        "helm/values/${ENVIRONMENT}/rabbitmq-values.yaml" \
        "10m"
    echo ""
    
    # Apply network policies
    echo -e "${YELLOW}🔒 Applying network policies...${NC}"
    kubectl apply -f k8s/data-stack/network-policies.yaml 2>/dev/null || true
    echo ""
    
    # Show completion info
    show_deployment_info
}

# Trap cleanup on script exit
trap cleanup EXIT

# Run main deployment
main_deploy "$@"8s/data-stack/clickhouse/values.yaml" ]; then
        echo -e "${RED}❌ ClickHouse values.yaml not found!${NC}"
        echo "Run: make init"
        exit 1
    fi
    
    if [ ! -f "k
