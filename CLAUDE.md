# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning research project implementing Snake game with fog of war mechanics. The agent can only see within a configurable field of view around its head, exploring partial observability in RL environments.

## Architecture

The project uses a Kubernetes-based modular architecture with separated services:

- **env**: Snake game environment server (Flask API on port 5000)
- **inference**: Agent inference service (connects to env via HTTP)
- **training**: Training service (gRPC server on port 50051) 
- **Data stack**: RabbitMQ (message queue) + ClickHouse (analytics database)

The current setup supports both Docker Compose (development) and Kubernetes (production) deployments.

### Service Structure
- `services/Env/`: Environment server (Flask-based Snake game)
- `services/Inference/`: Agent inference service
- `services/Training/`: Training service with gRPC interface
- `services/utils/`: Shared utilities
- `k8s/snake-rl/`: Helm chart for main RL application
- `k8s/data-stack/`: Data infrastructure (RabbitMQ, ClickHouse)

## Common Development Commands

### Data Stack Management (ClickHouse + RabbitMQ)
```bash
# Full setup from scratch
make quick-deploy

# Individual steps
make init              # Create namespace and config files
make secrets          # Generate passwords and create K8s secrets  
make deploy           # Deploy ClickHouse and RabbitMQ
make create-tables    # Initialize database tables
make status           # Check deployment status
make health           # Run health checks
make how-to-connect   # Get connection details
make destroy          # Remove everything (requires 'DELETE' confirmation)
```

### Development Workflows

#### Docker Compose with Shared Memory (Recommended)

**Production-ready setup** with Clock/Env/Inference services using POSIX shared memory:

```bash
cd services/

# 1. Build images (one-time setup)
docker build -t korolaab/snake_rl_base:latest -f Dockerfile .
docker build -t localhost:5000/snake-rl/clock:latest -f Clock/Dockerfile Clock/
docker build -t localhost:5000/snake-rl/env:latest -f Env/Dockerfile Env/
docker build -t localhost:5000/snake-rl/inference:latest -f Inference/Dockerfile Inference/

# 2. Run experiment (3000 episodes with MLflow tracking)
docker compose up

# 3. View results
cd logs-storage/ && mlflow ui --backend-store-uri file:///logs/mlruns
```

See `services/QUICKSTART.md` for detailed guide.

#### Docker Compose - Legacy HTTP/gRPC (Root Directory)

**Older architecture** with HTTP API and gRPC communication:

```bash
# Start experiment with interactive setup
./start_experiment.sh

# Manual Docker Compose
docker compose up --build
```

#### Kubernetes Deployment
```bash
# Deploy with Helm
helm install snake-rl k8s/snake-rl/ -n experiments --values k8s/snake-rl/values.yaml

# Build and push images
./scripts/build-and-push.sh

# Deploy using scripts
./scripts/deploy.sh
```

## Configuration

### Environment Variables
Key environment variables for services:
- `EXPERIMENT_NAME`: Experiment identifier
- `RUN_ID`: Unique run identifier (timestamp + UUID)
- `RABBITMQ_HOST`: RabbitMQ hostname (default: rabbitmq.data-stack.svc.cluster.local)
- `RABBITMQ_USERNAME/PASSWORD`: RabbitMQ credentials (tech/tech for development)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, etc.)

### Kubernetes Configuration
- Main chart: `k8s/snake-rl/values.yaml`
- Data stack: `k8s/data-stack/`
- Storage classes: `k8s/storage/`
- Namespaces: `experiments` (RL services), `data-stack` (infrastructure)

### Agent/Environment Parameters
Configure in `k8s/snake-rl/values.yaml`:
- `env.args`: Game parameters (fps, grid size, vision radius, rewards)
- `inference.args`: Agent parameters (learning rate, episodes, batch size)
- `inference.runAsJob`: true for Job, false for StatefulSet/Deployment

## Repository Structure Notes

- `legacy/`: Previous monolithic implementation (reference only)
- `helm/`: Helm charts for infrastructure
- `scripts/`: Deployment and utility scripts
- `shared/`: Shared configurations and volumes
- Current branch: `bug/bad_training` (branched from `hyperopt_tuning`)

## Testing and Development

The project is currently in active development focusing on:
- Debugging the current Kubernetes setup
- Expanding from 2 to 3 containers (env, inference, training)
- Adding support for multiple agents
- Agent collaboration and communication research

## Data Infrastructure

ClickHouse and RabbitMQ are managed via the data-stack namespace. Connection details and credentials are managed through Kubernetes secrets. Use `make how-to-connect` to get current connection information after deployment.