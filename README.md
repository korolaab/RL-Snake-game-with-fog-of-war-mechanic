# Snake RL with fog of war

## Abstract

This project is built for reinforcement learning (RL) research using a variant of the classic Snake game. The key feature is the introduction of a **fog of war** mechanism. Unlike the standard Snake, the agent (the "snake") is only able to see within a configurable field of view (FOV), a certain radius around its head. This setup is aimed at exploring partial observability in RL.

## Progress

1. **Monolithic Setup (legacy/):**  
   Initially, the environment and agent were combined into a single monolithic application. While this version allowed for the successful training of one agent, it was inflexible and made running experiments difficult.
   Results in [./legacy/Readme.md](./legacy/README.md)

2. **Kubernetes-Based Modular Setup:**  
   The architecture was refactored to Kubernetes distributed architecture with separeted blocks. The current setup consists of 2 containers:
   - **env:** The Snake game environment.
   - **inf:** The inference (agent) service + inside a training.  


3. **Next Steps:**  
   - Debug current setup
   - Expand to 3 containers: environment, inference, and training.
   - Add support for two agents in the same environment.
   - Research how agents can collaborate and communicate.

## How to Run

- **From Scratch:**  
  Follow the setup steps provided earlier in this README.

- **Using Existing Data Stack:**  
  If you already have a data stack available (RabbitMQ, PVC for logs), you can use it by updating the configuration:
    - Edit the `rabbitmq` hostname in `k8s/snake-rl/values.yaml` to point to your RabbitMQ instance.
    - Alternatively, configure log dumping to a file in your persistent volume claim (PVC) if preferred.

Please refer to the instructions in `/k8s/snake-rl/values.yaml` for further customization.

## Cold Setup

To get your RabbitMQ + ClickHouse data-stack running:

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
