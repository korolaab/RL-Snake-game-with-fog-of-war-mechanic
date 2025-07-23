# RL Distributed Learning Platform – Project Overview

This project is a modular, distributed reinforcement learning (RL) system that enables scalable training, real-time inference, and research experimentation on agents in a simulated environment. It is designed for flexibility, extensibility, and deployment in research or production settings using Docker and Kubernetes.
(For legacy data infrastructure setup, see "Cold Setup" below.)

---

## Key Components

- **services/**
  - `Inference/` – RL agent inference and environment control, REST & gRPC enabled.
  - `Training/` – Batched RL policy training using REINFORCE, model update via gRPC.
  - `Env/` (real or `mock_env/`) – The environment service. Exposes REST API for game state streaming and accepting actions. The real environment should use the same API as mock_env, and is typically deployed in production/test clusters.
  - `Sync/` – UDP synchronizer (optional; enables stepwise env/agent sync for deterministic RL experiments).
  - `mock_env/` – Mock Flask-based environment for testing agent logic without real backend (interchangeable with Env in standalone/dev mode).
  - `utils/` – Shared logger, protobufs, gRPC codegen utilities.
- **shared/** – Shared resources/configs.
- **k8s/** – Kubernetes manifests for service deployment.
- **examples/**, **scripts/** – Example pipelines, automation scripts, helpers.

## Project Features

- **Distributed Modular RL**: Each block (inference, training, environment) is scalable and isolated as its own service/docker image.
- **REST & gRPC APIs**: Inference communicates with Env by REST and with training service by gRPC/batch model update.
- **Pluggable Environment**: Swap in mock_env for local dev, or deploy the real Env which should conform to the same REST interface.
- **Synchronous & Asynchronous Modes**: Can run in async mode (default) or step-for-step mode using Sync service.
- **Extensive Logging**: Unified JSON logger supports local and message-queue-based monitoring (see utils/logger.py).
- **Easy Testing**: Includes `mock_env` to debug Inference quickly and locally.
- **Kubernetes & Docker Ready**: All services have Dockerfiles and are intended for orchestration in k8s clusters or rapid local dev.

## Service Overviews

- **Env** (or **mock_env**) – Exposes REST API used by the RL agent; streams states and receives commands.
- **Inference** – Receives states (REST), predicts actions (REST), batches experience, requests model updates (gRPC).
- **Training** – Receives model & experience batch (gRPC), runs policy updates, returns updated model (gRPC).
- **Sync** – UDP signal for coordinating stepwise progression among agents and environment (optional).
- **utils** – Logging, protobuf, and interface definitions (shared by all services).

## Development & Extensibility

- All agent, environment, model, and communication logic is Python (Torch, Flask, grpcio).
- Common Docker base image (`snake_rl_base`) ensures all dependencies available.
- Extend agents, environment logic, model architectures, or training methods as needed – see README.md of each core service.

---

## Cold Setup

Starting from scratch? Here's how to get your ClickHouse data stack running:

### Prerequisites

- kubectl configured for your cluster
- Helm 3 installed
- Access to pull from Docker Hub (bitnamicharts)

### Quick Setup (All-in-One)

If you want to run all steps automatically:

```bash
make quick-deploy
```

This runs init, secrets, deploy, health check, and create-tables in sequence.

### What Gets Deployed

- **ClickHouse**: OLAP database for analytics
- **RabbitMQ**: Message queue for data ingestion
- **Persistent volumes**: For data storage
- **Kubernetes secrets**: For secure credential management

### Getting Connection Details

After deployment, get connection information:

```bash
make how-to-connect
```

This shows URLs, credentials, and port-forwarding instructions for accessing your services.

### Troubleshooting

- Check deployment status: `make status`
- View service logs: `make logs`
- Run health checks: `make health`
- View passwords: `make passwords`

### Cleanup

To completely remove everything (WARNING: destroys all data):

```bash
make destroy
```

You'll need to type 'DELETE' to confirm.

---

*For complete instructions on each service, see the README.md inside `services/<ServiceName>/`.*
