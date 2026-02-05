# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning research project implementing Snake game with fog of war mechanics. The agent can only see within a configurable field of view around its head, exploring partial observability in RL environments.

## Architecture

The project uses a shared memory architecture with three services communicating via POSIX IPC:

- **clock**: Master coordinator managing execution timing and synchronization
- **env**: Snake game environment server (game logic and state)
- **inference**: Agent inference service (neural network and training)

All services communicate via shared memory (`/dev/shm`) using semaphores for synchronization. The current setup supports both Docker Compose (development) and Kubernetes (production) deployments.

### Service Structure
- `services/Clock/`: Coordination service (master orchestrator)
- `services/Env/`: Environment server (Snake game logic)
- `services/Inference/`: Agent inference service (neural network)
- `services/utils/`: Shared utilities
- `k8s/snake-rl/`: Helm chart for main RL application
- `k8s/minikube/`: Minikube setup scripts

## Common Development Commands

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
- `LOG_LEVEL`: Logging level (DEBUG, INFO, etc.)
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI

### Kubernetes Configuration
- Main chart: `k8s/snake-rl/values.yaml`
- Minikube setup: `k8s/minikube/setup.sh`
- Namespace: `default` (or as configured)

### Agent/Environment Parameters
Configure in `k8s/snake-rl/values.yaml` or `services/docker-compose.yaml`:
- `clock.args`: Coordination parameters (max episodes, fps, vision size)
- `env.args`: Game parameters (grid size, rewards, max lifetime)
- `inference.args`: Agent parameters (learning rate, gamma, beta)

## Repository Structure Notes

- `legacy/`: Previous monolithic implementation with Pygame (reference only, includes results)
- `docs/`: Documentation (ARCHITECTURE.md, PARAMETERS.md, HISTORY.md)
- `scripts/`: Deployment and utility scripts
- `services/`: Service implementations (Clock, Env, Inference)

## Testing and Development

The project uses shared memory architecture for fast, modular RL training:
- Clock service orchestrates execution timing
- All services communicate via POSIX IPC (shared memory + semaphores)
- MLflow tracking for experiment monitoring
- Docker Compose for local development, Kubernetes for production