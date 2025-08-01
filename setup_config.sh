#!/bin/bash

# ClickHouse Data Stack - Configuration Files Setup
# Creates all configuration files WITHOUT credentials for version control
# Run this once to set up your project structure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}📁 Setting up ClickHouse Data Stack Configuration Files${NC}"
echo -e "${BLUE}Creating files safe for version control (no credentials)${NC}"
echo ""

# Function to create directories
create_directories() {
    echo -e "${YELLOW}📁 Creating directory structure...${NC}"
    mkdir -p {k8s/{namespaces,storage,data-stack/{clickhouse/config,rabbitmq/config},experiments,monitoring},scripts,helm/values/{dev,staging,production},docs}
    echo "  ✓ k8s/ structure created"
    echo "  ✓ scripts/ directory created"
    echo "  ✓ helm/values/ for environments created"
    echo "  ✓ docs/ directory created"
}

# Function to create .gitignore
create_gitignore() {
    echo -e "${YELLOW}🚫 Creating .gitignore...${NC}"
    
    cat > .gitignore-data-stack << 'EOF'
# Secrets and credentials
secrets/
*.secret
*-secret.yaml
.env
.env.local

# Helm charts cache
charts/
*.tgz

# Kubernetes temporary files
*.tmp
kubeconfig*

# Local development
port-forward.log
*.pid

# Backup files
*.backup
backup/

# OS generated files
.DS_Store
Thumbs.db

# IDE files
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/
EOF

    echo "  ✓ .gitignore-data-stack created (merge with your main .gitignore)"
}

# Function to create namespace files
create_namespaces() {
    echo -e "${YELLOW}📝 Creating namespace definitions...${NC}"
    
    cat > k8s/namespaces/data-stack.yaml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: data-stack
  labels:
    name: data-stack
    purpose: data-services
    environment: shared
  annotations:
    description: "Persistent data services (ClickHouse, RabbitMQ)"
EOF

    cat > k8s/namespaces/experiments.yaml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: experiments
  labels:
    name: experiments
    purpose: rl-ml-workloads
    environment: dynamic
  annotations:
    description: "RL training jobs and ML experiments"
EOF

    cat > k8s/namespaces/monitoring.yaml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
  labels:
    name: monitoring
    purpose: observability
    environment: shared
  annotations:
    description: "Monitoring stack (Prometheus, Grafana)"
EOF

    echo "  ✓ Namespace definitions created"
}

# Function to create storage configurations
create_storage_configs() {
    echo -e "${YELLOW}💾 Creating storage configurations...${NC}"
    
    cat > k8s/storage/local-path-storage-class.yaml << 'EOF'
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: local-path
  annotations:
    storageclass.kubernetes.io/is-default-class: "false"
    description: "Local path storage for development"
provisioner: rancher.io/local-path
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Delete
allowVolumeExpansion: true
parameters:
  hostPath: "/opt/local-path-provisioner"
EOF

    cat > k8s/storage/fast-ssd-storage-class.yaml << 'EOF'
# Example for production SSD storage
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
  annotations:
    description: "Fast SSD storage for production workloads"
# Uncomment and configure for your cloud provider:
# AWS EKS:
# provisioner: kubernetes.io/aws-ebs
# parameters:
#   type: gp3
#   fsType: ext4

# GKE:
# provisioner: kubernetes.io/gce-pd
# parameters:
#   type: pd-ssd

# Azure:
# provisioner: kubernetes.io/azure-disk
# parameters:
#   storageaccounttype: Premium_LRS
#   kind: Managed

provisioner: kubernetes.io/no-provisioner  # Replace with your provider
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Retain
allowVolumeExpansion: true
EOF

    echo "  ✓ Storage classes created"
}

# Function to create ClickHouse values template
create_clickhouse_values() {
    echo -e "${YELLOW}🗃️  Creating ClickHouse values templates...${NC}"
    
    cat > k8s/data-stack/clickhouse/values-template.yaml << 'EOF'
# ClickHouse Helm Values Template
# Copy to values.yaml and customize with your credentials

global:
  storageClass: "local-path"

auth:
  # TODO: Set password in values.yaml (not committed)
  password: ""
  # Or use existing secret:
  # existingSecret: "clickhouse-credentials"
  # existingSecretKey: "password"

persistence:
  enabled: true
  storageClass: "local-path"  # Change for production
  size: 50Gi
  accessModes:
    - ReadWriteOnce

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

service:
  type: ClusterIP
  ports:
    http: 8123
    tcp: 9000
  annotations: {}

metrics:
  enabled: true
  serviceMonitor:
    enabled: false  # Enable if using Prometheus Operator

# ZooKeeper for ClickHouse clustering
zookeeper:
  enabled: true
  persistence:
    enabled: true
    size: 8Gi
    storageClass: "local-path"
  resources:
    requests:
      cpu: 250m
      memory: 256Mi
    limits:
      cpu: 500m
      memory: 512Mi

# Clustering configuration
shards: 1
replicaCount: 1

# Custom ClickHouse configuration
configuration: |
  <clickhouse>
    <logger>
      <level>information</level>
      <console>true</console>
    </logger>
    <http_port>8123</http_port>
    <tcp_port>9000</tcp_port>
    <interserver_http_port>9009</interserver_http_port>
    <listen_host>::</listen_host>
    <max_connections>4096</max_connections>
    <keep_alive_timeout>3</keep_alive_timeout>
    <max_concurrent_queries>100</max_concurrent_queries>
    <uncompressed_cache_size>8589934592</uncompressed_cache_size>
    <mark_cache_size>5368709120</mark_cache_size>
    <timezone>UTC</timezone>
  </clickhouse>

# User profiles and quotas
users: |
  <clickhouse>
    <users>
      <default>
        <networks>
          <ip>::/0</ip>
        </networks>
        <profile>default</profile>
        <quota>default</quota>
      </default>
    </users>
    <profiles>
      <default>
        <max_memory_usage>10000000000</max_memory_usage>
        <use_uncompressed_cache>0</use_uncompressed_cache>
        <load_balancing>random</load_balancing>
        <max_execution_time>300</max_execution_time>
      </default>
    </profiles>
    <quotas>
      <default>
        <interval>
          <duration>3600</duration>
          <queries>0</queries>
          <errors>0</errors>
          <result_rows>0</result_rows>
          <read_rows>0</read_rows>
          <execution_time>0</execution_time>
        </interval>
      </default>
    </quotas>
  </clickhouse>

# Pod security context
podSecurityContext:
  fsGroup: 1001

# Container security context
containerSecurityContext:
  runAsUser: 1001
  runAsNonRoot: true
  readOnlyRootFilesystem: false

# Node selector and tolerations
nodeSelector: {}
tolerations: []
affinity: {}
EOF

    # Environment-specific values
    cat > helm/values/dev/clickhouse-values.yaml << 'EOF'
# Development environment overrides
persistence:
  size: 10Gi

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 250m
    memory: 512Mi

zookeeper:
  persistence:
    size: 2Gi
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 250m
      memory: 256Mi
EOF

    cat > helm/values/production/clickhouse-values.yaml << 'EOF'
# Production environment overrides
persistence:
  storageClass: "fast-ssd"
  size: 500Gi

resources:
  limits:
    cpu: 8000m
    memory: 16Gi
  requests:
    cpu: 2000m
    memory: 8Gi

replicaCount: 3
shards: 2

service:
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"

zookeeper:
  replicaCount: 3
  persistence:
    storageClass: "fast-ssd"
    size: 20Gi

podAntiAffinity:
  preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchLabels:
            app.kubernetes.io/name: clickhouse
        topologyKey: kubernetes.io/hostname
EOF

    echo "  ✓ ClickHouse value templates created"
}

# Function to create RabbitMQ values template
create_rabbitmq_values() {
    echo -e "${YELLOW}🐰 Creating RabbitMQ values templates...${NC}"
    
    cat > k8s/data-stack/rabbitmq/values-template.yaml << 'EOF'
# RabbitMQ Helm Values Template
# Copy to values.yaml and customize with your credentials

auth:
  username: admin
  # TODO: Set password in values.yaml (not committed)
  password: ""
  # Or use existing secret:
  # existingPasswordSecret: "rabbitmq-credentials"

persistence:
  enabled: true
  storageClass: "local-path"
  size: 8Gi
  accessModes:
    - ReadWriteOnce

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 250m
    memory: 512Mi

service:
  type: ClusterIP
  ports:
    amqp: 5672
    amqpTls: 5671
    dist: 25672
    manager: 15672
    epmd: 4369

metrics:
  enabled: true
  serviceMonitor:
    enabled: false

# Clustering (disable for single-node)
clustering:
  enabled: false
  rebalance: false

# Memory and disk limits
memoryHighWatermark:
  enabled: true
  type: relative
  value: 0.4

diskFreeLimit:
  enabled: true
  absolute: 2GB

# Plugins
plugins: "rabbitmq_management rabbitmq_prometheus rabbitmq_shovel rabbitmq_shovel_management"

# Extra configuration
extraConfiguration: |
  management.tcp.port = 15672
  prometheus.tcp.port = 15692
  management.tcp.ip = 0.0.0.0
  loopback_users = none

# Community plugins (optional)
communityPlugins: ""

# Load definitions (exchanges, queues, etc.)
loadDefinition:
  enabled: false
  # secretName: "rabbitmq-load-definition"

# Pod security context
podSecurityContext:
  fsGroup: 1001

# Container security context
containerSecurityContext:
  runAsUser: 1001
  runAsNonRoot: true

# Node selector and tolerations
nodeSelector: {}
tolerations: []
affinity: {}
EOF

    cat > helm/values/dev/rabbitmq-values.yaml << 'EOF'
# Development environment overrides
persistence:
  size: 2Gi

resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 100m
    memory: 256Mi

clustering:
  enabled: false
EOF

    cat > helm/values/production/rabbitmq-values.yaml << 'EOF'
# Production environment overrides
persistence:
  storageClass: "fast-ssd"
  size: 50Gi

resources:
  limits:
    cpu: 4000m
    memory: 8Gi
  requests:
    cpu: 1000m
    memory: 4Gi

clustering:
  enabled: true
  replicaCount: 3

service:
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"

podAntiAffinity:
  preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchLabels:
            app.kubernetes.io/name: rabbitmq
        topologyKey: kubernetes.io/hostname
EOF

    echo "  ✓ RabbitMQ value templates created"
}

# Function to create network policies
create_network_policies() {
    echo -e "${YELLOW}🔒 Creating network policies...${NC}"
    
    cat > k8s/data-stack/network-policies.yaml << 'EOF'
# Network policies for data-stack namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: data-stack-ingress
  namespace: data-stack
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  ingress:
  # Allow traffic from experiments namespace
  - from:
    - namespaceSelector:
        matchLabels:
          name: experiments
  # Allow traffic within data-stack namespace
  - from:
    - namespaceSelector:
        matchLabels:
          name: data-stack
  # Allow traffic from monitoring namespace
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
  # Allow ingress controllers (if needed)
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: data-stack-egress
  namespace: data-stack
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  # Allow all egress (can be restricted further)
  - {}
---
# Allow experiments to access data-stack
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: experiments-to-data-stack
  namespace: experiments
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  # Allow access to data-stack
  - to:
    - namespaceSelector:
        matchLabels:
          name: data-stack
  # Allow DNS resolution
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  # Allow external traffic (internet, etc.)
  - to: []
    ports:
    - protocol: TCP
    - protocol: UDP
EOF

    echo "  ✓ Network policies created"
}

# Function to create scripts templates
create_script_templates() {
    echo -e "${YELLOW}📜 Creating script templates...${NC}"
    
    cat > scripts/deploy-template.sh << 'EOF'
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
EOF

    cat > scripts/port-forward-template.sh << 'EOF'
#!/bin/bash
# Port Forward Script Template
# Copy to port-forward.sh and customize

NAMESPACE="data-stack"

echo "🔗 Setting up port forwarding..."

# Check if services are running
check_service() {
    if ! kubectl get svc $1 -n ${NAMESPACE} >/dev/null 2>&1; then
        echo "❌ Service $1 not found in namespace ${NAMESPACE}"
        return 1
    fi
    return 0
}

# Port forward function
start_port_forward() {
    local service=$1
    local local_port=$2
    local remote_port=$3
    
    if check_service ${service}; then
        kubectl port-forward svc/${service} ${local_port}:${remote_port} -n ${NAMESPACE} &
        echo "✅ ${service}: localhost:${local_port}"
        return $!
    fi
}

# Start port forwards
start_port_forward "clickhouse" "8123" "8123"
CH_PID=$?

start_port_forward "rabbitmq" "15672" "15672"
RMQ_PID=$?

start_port_forward "rabbitmq" "5672" "5672"
RMQ_AMQP_PID=$?

echo ""
echo "🎯 Access services at:"
echo "  - ClickHouse: http://localhost:8123/play"
echo "  - RabbitMQ Management: http://localhost:15672"
echo ""
echo "Press Ctrl+C to stop"

# Cleanup
cleanup() {
    echo "🛑 Stopping port forwards..."
    jobs -p | xargs -r kill
}

trap cleanup EXIT INT TERM
wait
EOF

    chmod +x scripts/*.sh
    echo "  ✓ Script templates created"
}

# Function to create Makefile
create_makefile() {
    echo -e "${YELLOW}⚙️  Creating Makefile...${NC}"
    
    cat > Makefile << 'EOF'
.PHONY: help init deploy destroy status logs clean

# Configuration
NAMESPACE := data-stack
ENVIRONMENT ?= dev

help: ## Show this help
	@echo "ClickHouse Data Stack Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

init: ## Initialize configuration files from templates
	@echo "🔧 Initializing configuration files..."
	@if [ ! -f k8s/data-stack/clickhouse/values.yaml ]; then \
		cp k8s/data-stack/clickhouse/values-template.yaml k8s/data-stack/clickhouse/values.yaml; \
		echo "✅ Created ClickHouse values.yaml - please edit credentials"; \
	fi
	@if [ ! -f k8s/data-stack/rabbitmq/values.yaml ]; then \
		cp k8s/data-stack/rabbitmq/values-template.yaml k8s/data-stack/rabbitmq/values.yaml; \
		echo "✅ Created RabbitMQ values.yaml - please edit credentials"; \
	fi
	@if [ ! -f scripts/deploy.sh ]; then \
		cp scripts/deploy-template.sh scripts/deploy.sh; \
		chmod +x scripts/deploy.sh; \
		echo "✅ Created deploy.sh"; \
	fi
	@if [ ! -f scripts/port-forward.sh ]; then \
		cp scripts/port-forward-template.sh scripts/port-forward.sh; \
		chmod +x scripts/port-forward.sh; \
		echo "✅ Created port-forward.sh"; \
	fi

deploy: ## Deploy data stack
	@ENVIRONMENT=$(ENVIRONMENT) ./scripts/deploy.sh

destroy: ## Destroy data stack (WARNING: deletes data)
	@echo "⚠️  WARNING: This will delete all data in $(NAMESPACE)!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	@helm uninstall clickhouse -n $(NAMESPACE) 2>/dev/null || true
	@helm uninstall rabbitmq -n $(NAMESPACE) 2>/dev/null || true
	@kubectl delete namespace $(NAMESPACE) --ignore-not-found

status: ## Show status of all services
	@echo "📊 Data Stack Status:"
	@kubectl get pods -n $(NAMESPACE) 2>/dev/null || echo "Namespace $(NAMESPACE) not found"
	@echo "\n🔗 Services:"
	@kubectl get svc -n $(NAMESPACE) 2>/dev/null || true
	@echo "\n💾 Storage:"
	@kubectl get pvc -n $(NAMESPACE) 2>/dev/null || true

logs: ## Show logs from services
	@echo "📋 Recent logs:"
	@kubectl logs -l app.kubernetes.io/name=clickhouse -n $(NAMESPACE) --tail=20 2>/dev/null || true
	@kubectl logs -l app.kubernetes.io/name=rabbitmq -n $(NAMESPACE) --tail=20 2>/dev/null || true

port-forward: ## Start port forwarding
	@./scripts/port-forward.sh

clean: ## Clean up failed resources
	@kubectl delete pods --field-selector=status.phase=Failed -n $(NAMESPACE) 2>/dev/null || true
	@helm list -n $(NAMESPACE) --failed -q | xargs -r helm delete -n $(NAMESPACE)

dev: ## Quick deploy for development
	@$(MAKE) ENVIRONMENT=dev deploy

prod: ## Deploy for production
	@$(MAKE) ENVIRONMENT=production deploy
EOF

    echo "  ✓ Makefile created"
}

# Function to create documentation
create_documentation() {
    echo -e "${YELLOW}📖 Creating documentation...${NC}"
    
    cat > docs/SETUP.md << 'EOF'
# ClickHouse Data Stack Setup Guide

## Prerequisites

- Kubernetes cluster (minikube, kind, or cloud provider)
- Helm 3.x
- kubectl configured

## Initial Setup

1. **Run configuration setup** (one-time):
   ```bash
   ./setup-config.sh
   ```

2. **Initialize working files**:
   ```bash
   make init
   ```

3. **Set credentials** in generated files:
   - `k8s/data-stack/clickhouse/values.yaml`
   - `k8s/data-stack/rabbitmq/values.yaml`

4. **Deploy the stack**:
   ```bash
   make deploy
   ```

## File Structure

```
k8s/
├── namespaces/              # Kubernetes namespaces
├── storage/                 # Storage classes
└── data-stack/              # Service configurations
    ├── clickhouse/
    │   ├── values-template.yaml  # Template (committed)
    │   └── values.yaml          # Working file (not committed)
    └── rabbitmq/
        ├── values-template.yaml  # Template (committed)
        └── values.yaml          # Working file (not committed)

helm/values/                 # Environment-specific overrides
├── dev/
├── staging/
└── production/

scripts/                     # Deployment utilities
├── deploy-template.sh       # Template (committed)
├── deploy.sh               # Working file (not committed)
└── port-forward.sh         # Working file (not committed)
```

## Commands

```bash
make help           # Show all commands
make init           # Create working files from templates
make deploy         # Deploy with default environment (dev)
make dev            # Deploy development environment
make prod           # Deploy production environment
make status         # Check service status
make port-forward   # Access services locally
make destroy        # Remove everything (WARNING: deletes data)
```

## Environments

- **dev**: Minimal resources for local development
- **staging**: Production-like setup for testing
- **production**: Full resources with clustering

Set environment: `ENVIRONMENT=staging make deploy`

## Security Notes

- Templates contain no credentials
- Working files with credentials are gitignored
- Network policies restrict access between namespaces
- Use Kubernetes secrets for production credentials

## Troubleshooting

1. **Services not starting**: Check `make status` and `make logs`
2. **Port conflicts**: Stop existing port forwards
3. **Storage issues**: Verify storage class exists
4. **Permission errors**: Check pod security contexts
EOF

    cat > docs/USAGE.md << 'EOF'
# Using ClickHouse and RabbitMQ

## Accessing Services

### Local Development (Port Forward)

```bash
make port-forward
```

- **ClickHouse**: http://localhost:8123/play
- **RabbitMQ**: http://localhost:15672

### From Applications (In-Cluster)

**ClickHouse**:
- HTTP: `clickhouse.data-stack.svc.cluster.local:8123`
- TCP: `clickhouse.data-stack.svc.cluster.local:9000`

**RabbitMQ**:
- AMQP: `rabbitmq.data-stack.svc.cluster.local:5672`
- Management: `rabbitmq.data-stack.svc.cluster.local:15672`

## Python Examples

### ClickHouse

```python
import clickhouse_connect

client = clickhouse_connect.get_client(
    host='clickhouse.data-stack.svc.cluster.local',
    port=8123,
    username='default',
    password='your-password'
)

# Create table
client.execute("""
    CREATE TABLE IF NOT EXISTS rl_experiments (
        id UUID DEFAULT generateUUIDv4(),
        experiment_name String,
        episode Int32,
        reward Float64,
        timestamp DateTime DEFAULT now()
    ) ENGINE = MergeTree()
    ORDER BY (experiment_name, episode)
""")

# Insert data
client.insert('rl_experiments', [
    ['dqn_cartpole', 1, 200.0],
    ['dqn_cartpole', 2, 195.0]
], column_names=['experiment_name', 'episode', 'reward'])

# Query data
result = client.query('SELECT * FROM rl_experiments LIMIT 10')
print(result.result_rows)
```

### RabbitMQ

```python
import pika
import json

# Connection
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='rabbitmq.data-stack.svc.cluster.local',
        port=5672,
        credentials=pika.PlainCredentials('admin', 'your-password')
    )
)
channel = connection.channel()

# Declare queue
channel.queue_declare(queue='rl_jobs', durable=True)

# Publish job
job = {
    'algorithm': 'dqn',
    'environment': 'CartPole-v1',
    'episodes': 1000
}

channel.basic_publish(
    exchange='',
    routing_key='rl_jobs',
    body=json.dumps(job),
    properties=pika.BasicProperties(delivery_mode=2)  # Persistent
)

# Consume jobs
def process_job(ch, method, properties, body):
    job = json.loads(body)
    print(f"Processing job: {job}")
    # Your RL training logic here
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(queue='rl_jobs', on_message_callback=process_job)
channel.start_consuming()
```

## Common Patterns

### RL Experiment Logging

Store experiment metrics in ClickHouse for analysis:

```sql
-- Create experiments table
CREATE TABLE experiments (
    experiment_id UUID,
    name String,
    algorithm String,
    environment String,
    episode Int32,
    step Int32,
    reward Float64,
    loss Float64,
    epsilon Float64,
    timestamp DateTime
) ENGINE = MergeTree()
ORDER BY (experiment_id, episode, step);

-- Analyze performance
SELECT 
    name,
    algorithm,
    AVG(reward) as avg_reward,
    MAX(reward) as max_reward
FROM experiments 
WHERE episode > 100  -- Skip initial episodes
GROUP BY name, algorithm
ORDER BY avg_reward DESC;
```

### Job Queue Management

Use RabbitMQ for distributing training jobs:

```python
# Job producer
def submit_training_job(algorithm, env, params):
    job = {
        'id': str(uuid.uuid4()),
        'algorithm': algorithm,
        'environment': env,
        'parameters': params,
        'created_at': datetime.now().isoformat()
    }
    
    channel.basic_publish(
        exchange='training',
        routing_key=f'jobs.{algorithm}',
        body=json.dumps(job)
    )

# Job consumer
def train_model(job_data):
    # Load environment
    env = gym.make(job_data['environment'])
    
    # Initialize algorithm
    if job_data['algorithm'] == 'dqn':
        agent = DQNAgent(env, **job_data['parameters'])
    
    # Train and log to ClickHouse
    for episode in range(job_data.get('episodes', 1000)):
        reward = agent.train_episode()
        log_to_clickhouse(job_data['id'], episode, reward)
```
EOF

    cat > docs/ENVIRONMENTS.md << 'EOF'
# Environment Configuration

## Development Environment

For local development and testing:

```yaml
# helm/values/dev/clickhouse-values.yaml
persistence:
  size: 10Gi
  storageClass: "local-path"

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 250m
    memory: 512Mi

shards: 1
replicaCount: 1
```

## Staging Environment

Production-like setup for testing:

```yaml
# helm/values/staging/clickhouse-values.yaml
persistence:
  size: 100Gi
  storageClass: "fast-ssd"

resources:
  limits:
    cpu: 4000m
    memory: 8Gi
  requests:
    cpu: 1000m
    memory: 4Gi

shards: 2
replicaCount: 2
```

## Production Environment

Full production setup with clustering:

```yaml
# helm/values/production/clickhouse-values.yaml
persistence:
  size: 500Gi
  storageClass: "fast-ssd"

resources:
  limits:
    cpu: 8000m
    memory: 16Gi
  requests:
    cpu: 2000m
    memory: 8Gi

shards: 3
replicaCount: 3

# High availability
podAntiAffinity:
  preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchLabels:
            app.kubernetes.io/name: clickhouse
        topologyKey: kubernetes.io/hostname
```

## Cloud Provider Configurations

### AWS EKS

```yaml
# Storage class for AWS
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3-encrypted
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
  fsType: ext4
volumeBindingMode: WaitForFirstConsumer
```

### Google GKE

```yaml
# Storage class for GCP
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ssd-encrypted
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  replication-type: regional-pd
volumeBindingMode: WaitForFirstConsumer
```

### Azure AKS

```yaml
# Storage class for Azure
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: premium-ssd
provisioner: kubernetes.io/azure-disk
parameters:
  storageaccounttype: Premium_LRS
  kind: Managed
volumeBindingMode: WaitForFirstConsumer
```
EOF

    echo "  ✓ Documentation created"
}

# Function to create example configurations
create_examples() {
    echo -e "${YELLOW}💡 Creating example configurations...${NC}"
    
    mkdir -p examples/{python,kubernetes,jupyter}
    
    cat > examples/python/clickhouse_example.py << 'EOF'
"""
Example: Using ClickHouse for RL experiment logging
"""
import clickhouse_connect
import uuid
from datetime import datetime

class ExperimentLogger:
    def __init__(self, host='localhost', port=8123, username='default', password=''):
        self.client = clickhouse_connect.get_client(
            host=host, port=port, username=username, password=password
        )
        self.setup_tables()
    
    def setup_tables(self):
        """Create necessary tables for RL experiments"""
        self.client.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id UUID,
                name String,
                algorithm String,
                environment String,
                hyperparameters String,  -- JSON string
                created_at DateTime
            ) ENGINE = MergeTree()
            ORDER BY (experiment_id, created_at)
        """)
        
        self.client.execute("""
            CREATE TABLE IF NOT EXISTS episode_metrics (
                experiment_id UUID,
                episode Int32,
                total_reward Float64,
                episode_length Int32,
                loss Float64,
                epsilon Float64,
                timestamp DateTime
            ) ENGINE = MergeTree()
            ORDER BY (experiment_id, episode)
        """)
    
    def start_experiment(self, name, algorithm, environment, hyperparameters):
        """Start a new experiment"""
        experiment_id = str(uuid.uuid4())
        
        self.client.insert('experiments', [[
            experiment_id, name, algorithm, environment,
            str(hyperparameters), datetime.now()
        ]], column_names=[
            'experiment_id', 'name', 'algorithm', 'environment',
            'hyperparameters', 'created_at'
        ])
        
        return experiment_id
    
    def log_episode(self, experiment_id, episode, reward, length, loss=None, epsilon=None):
        """Log metrics for a single episode"""
        self.client.insert('episode_metrics', [[
            experiment_id, episode, reward, length,
            loss or 0.0, epsilon or 0.0, datetime.now()
        ]], column_names=[
            'experiment_id', 'episode', 'total_reward', 'episode_length',
            'loss', 'epsilon', 'timestamp'
        ])
    
    def get_experiment_stats(self, experiment_id):
        """Get statistics for an experiment"""
        return self.client.query(f"""
            SELECT 
                COUNT(*) as episodes,
                AVG(total_reward) as avg_reward,
                MAX(total_reward) as max_reward,
                AVG(episode_length) as avg_length
            FROM episode_metrics 
            WHERE experiment_id = '{experiment_id}'
        """).result_rows[0]

# Usage example
if __name__ == "__main__":
    logger = ExperimentLogger()
    
    # Start experiment
    exp_id = logger.start_experiment(
        name="DQN CartPole Test",
        algorithm="DQN",
        environment="CartPole-v1",
        hyperparameters={"lr": 0.001, "batch_size": 32}
    )
    
    # Simulate logging episodes
    for episode in range(100):
        reward = 200 - episode * 0.5  # Simulated improving performance
        logger.log_episode(exp_id, episode, reward, 200)
    
    # Get stats
    stats = logger.get_experiment_stats(exp_id)
    print(f"Experiment stats: {stats}")
EOF

    cat > examples/python/rabbitmq_example.py << 'EOF'
"""
Example: Using RabbitMQ for RL job distribution
"""
import pika
import json
import uuid
from datetime import datetime

class RLJobQueue:
    def __init__(self, host='localhost', port=5672, username='admin', password=''):
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host, port=port, credentials=credentials)
        )
        self.channel = self.connection.channel()
        self.setup_queues()
    
    def setup_queues(self):
        """Setup queues and exchanges for RL jobs"""
        # Training jobs queue
        self.channel.queue_declare(queue='training_jobs', durable=True)
        
        # Results queue
        self.channel.queue_declare(queue='training_results', durable=True)
        
        # Priority queue for urgent jobs
        self.channel.queue_declare(
            queue='priority_jobs', 
            durable=True,
            arguments={'x-max-priority': 10}
        )
    
    def submit_training_job(self, algorithm, environment, hyperparameters, priority=0):
        """Submit a training job to the queue"""
        job = {
            'job_id': str(uuid.uuid4()),
            'algorithm': algorithm,
            'environment': environment,
            'hyperparameters': hyperparameters,
            'submitted_at': datetime.now().isoformat(),
            'status': 'queued'
        }
        
        queue = 'priority_jobs' if priority > 0 else 'training_jobs'
        
        self.channel.basic_publish(
            exchange='',
            routing_key=queue,
            body=json.dumps(job),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                priority=priority
            )
        )
        
        return job['job_id']
    
    def submit_result(self, job_id, metrics):
        """Submit training results"""
        result = {
            'job_id': job_id,
            'metrics': metrics,
            'completed_at': datetime.now().isoformat()
        }
        
        self.channel.basic_publish(
            exchange='',
            routing_key='training_results',
            body=json.dumps(result),
            properties=pika.BasicProperties(delivery_mode=2)
        )
    
    def process_jobs(self, callback):
        """Process jobs from the queue"""
        def wrapper(ch, method, properties, body):
            job = json.loads(body)
            try:
                result = callback(job)
                if result:
                    self.submit_result(job['job_id'], result)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                print(f"Job failed: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='training_jobs', on_message_callback=wrapper)
        self.channel.basic_consume(queue='priority_jobs', on_message_callback=wrapper)
        
        print("Waiting for jobs. To exit press CTRL+C")
        self.channel.start_consuming()

# Usage example
def train_agent(job):
    """Example training function"""
    print(f"Training {job['algorithm']} on {job['environment']}")
    
    # Simulate training
    import time
    time.sleep(5)  # Simulate training time
    
    # Return metrics
    return {
        'avg_reward': 195.5,
        'episodes': 1000,
        'training_time': 300
    }

if __name__ == "__main__":
    queue = RLJobQueue()
    
    # Submit some jobs
    job1 = queue.submit_training_job("DQN", "CartPole-v1", {"lr": 0.001})
    job2 = queue.submit_training_job("PPO", "CartPole-v1", {"lr": 0.0003}, priority=5)
    
    print(f"Submitted jobs: {job1}, {job2}")
    
    # Process jobs (in a real scenario, this would be in a separate worker)
    # queue.process_jobs(train_agent)
EOF

    cat > examples/kubernetes/rl-training-job.yaml << 'EOF'
# Example Kubernetes Job for RL Training
apiVersion: batch/v1
kind: Job
metadata:
  name: dqn-cartpole-training
  namespace: experiments
spec:
  template:
    spec:
      containers:
      - name: rl-trainer
        image: your-registry/rl-trainer:latest
        env:
        - name: CLICKHOUSE_HOST
          value: "clickhouse.data-stack.svc.cluster.local"
        - name: CLICKHOUSE_PORT
          value: "8123"
        - name: RABBITMQ_HOST
          value: "rabbitmq.data-stack.svc.cluster.local"
        - name: RABBITMQ_PORT
          value: "5672"
        - name: ALGORITHM
          value: "DQN"
        - name: ENVIRONMENT
          value: "CartPole-v1"
        - name: EPISODES
          value: "1000"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      restartPolicy: Never
  backoffLimit: 3
---
# PVC for model storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage-pvc
  namespace: experiments
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: local-path
EOF

    echo "  ✓ Example configurations created"
}

# Function to create final summary
create_summary() {
    echo -e "${YELLOW}📋 Creating setup summary...${NC}"
    
    cat > SETUP-SUMMARY.md << 'EOF'
# ClickHouse Data Stack - Setup Complete

## ✅ Files Created

### Configuration Templates (Safe for Git)
- `k8s/namespaces/` - Kubernetes namespace definitions
- `k8s/storage/` - Storage class configurations
- `k8s/data-stack/clickhouse/values-template.yaml` - ClickHouse template
- `k8s/data-stack/rabbitmq/values-template.yaml` - RabbitMQ template
- `k8s/data-stack/network-policies.yaml` - Network security policies
- `helm/values/{dev,staging,production}/` - Environment-specific configs

### Scripts and Automation
- `scripts/deploy-template.sh` - Deployment script template
- `scripts/port-forward-template.sh` - Port forwarding template
- `Makefile` - Build automation

### Documentation
- `docs/SETUP.md` - Complete setup guide
- `docs/USAGE.md` - Usage examples and patterns
- `docs/ENVIRONMENTS.md` - Environment configurations
- `examples/` - Python and Kubernetes examples

### Git Configuration
- `.gitignore-data-stack` - Git ignore rules for secrets

## 🚀 Next Steps

1. **Initialize working files**:
   ```bash
   make init
   ```

2. **Set credentials** in:
   - `k8s/data-stack/clickhouse/values.yaml`
   - `k8s/data-stack/rabbitmq/values.yaml`

3. **Deploy the stack**:
   ```bash
   make deploy
   ```

4. **Access services locally**:
   ```bash
   make port-forward
   ```

## 🔧 Quick Commands

```bash
make help           # Show all available commands
make init           # Create working files from templates
make deploy         # Deploy data stack
make status         # Check service status
make port-forward   # Access services locally
make destroy        # Remove everything (WARNING: deletes data)
```

## 📚 Learn More

- Read `docs/SETUP.md` for detailed instructions
- Check `docs/USAGE.md` for integration examples
- Review `examples/` for code samples
- Examine `docs/ENVIRONMENTS.md` for production setup

## 🔒 Security Notes

- Templates contain NO credentials
- Working files with credentials are automatically gitignored
- Network policies restrict inter-namespace communication
- Use Kubernetes secrets for production deployments

---

**Happy coding with your RL data stack! 🤖📊**
EOF

    echo "  ✓ Setup summary created"
}

# Main execution function
main() {
    echo -e "${BLUE}🏗️  Setting up ClickHouse Data Stack configuration files...${NC}"
    echo -e "${BLUE}This creates templates and structure WITHOUT credentials${NC}"
    echo ""
    
    create_directories
    create_gitignore
    create_namespaces
    create_storage_configs
    create_clickhouse_values
    create_rabbitmq_values
    create_network_policies
    create_script_templates
    create_makefile
    create_documentation
    create_examples
    create_summary
    
    echo ""
    echo -e "${GREEN}🎉 Configuration setup completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}📋 What was created:${NC}"
    echo "  ✓ Complete directory structure"
    echo "  ✓ Configuration templates (no credentials)"
    echo "  ✓ Deployment scripts and Makefile"
    echo "  ✓ Documentation and examples"
    echo "  ✓ Git ignore rules for secrets"
    echo ""
    echo -e "${GREEN}🚀 Next steps:${NC}"
    echo "  1. Run: ${BLUE}make init${NC} (creates working files)"
    echo "  2. Edit credentials in generated values.yaml files"
    echo "  3. Run: ${BLUE}make deploy${NC} (deploys everything)"
    echo "  4. Run: ${BLUE}make port-forward${NC} (local access)"
    echo ""
    echo -e "${BLUE}📖 Read SETUP-SUMMARY.md for complete instructions${NC}"
}

# Run main function
main "$@"
