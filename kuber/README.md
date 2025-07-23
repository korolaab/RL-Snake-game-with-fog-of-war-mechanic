# RL Distributed Learning Platform – Project Overview

This project is a modular, distributed reinforcement learning (RL) system that enables scalable training, real-time inference, and research experimentation on agents in a simulated environment. It is designed for flexibility, extensibility, and deployment in research or production settings using Docker and Kubernetes.
(For legacy data infrastructure setup, see "Cold Setup" below.)

---

## Key Components

- **services/**
  - `Inference/` – RL agent inference and environment control, REST & gRPC enabled.
  - `Training/` – Batched RL policy training using REINFORCE, model update via gRPC.
  - `Sync/` – UDP synchronizer (optional; enables stepwise env/agent sync for deterministic RL experiments).
  - `mock_env/` – Mock Flask-based environment for testing agent logic without real backend.
  - `utils/` – Shared logger, protobufs, gRPC codegen utilities.
- **shared/** – Shared resources/configs.
- **k8s/** – Kubernetes manifests for service deployment.
- **examples/**, **scripts/** – Example pipelines, automation scripts, helpers.

## Project Features

- **Distributed Modular RL**: Each block (inference, training, environment) is scalable and isolated as its own service/docker image.
- **REST & gRPC APIs**: Inference communicates with env by REST and training service by gRPC/batch model update.
- **Synchronous & Asynchronous Modes**: Can run in async mode (default) or step-for-step mode using Sync service.
- **Extensive Logging**: Unified JSON logger supports local and message-queue-based monitoring (see utils/logger.py).
- **Easy Testing**: Includes `mock_env` to debug Inference quickly and locally.
- **Kubernetes & Docker Ready**: All services have Dockerfiles and are intended for orchestration in k8s clusters or rapid local dev.

## How to Use

1. **Build core images**
   - From project root or each service directory:
     ```sh
     docker build -t <service_name> ./services/<ServiceName>
