# Snake RL Helm Chart

Kubernetes Helm chart for running Snake RL experiments with shared memory architecture.

## Quick Start (Minikube)

```bash
# Run automated setup
cd ../minikube
./setup.sh

# Watch progress
kubectl get pods -w

# View logs
kubectl logs job/snake-rl -c clock -f
```

## Manual Installation

### 1. Build Docker Images

```bash
cd ../../services

# Build base image
docker build -t korolaab/snake_rl_base:latest -f Dockerfile .

# Build service images
docker build -t localhost:5000/snake-rl/clock:latest -f Clock/Dockerfile Clock/
docker build -t localhost:5000/snake-rl/env:latest -f Env/Dockerfile Env/
docker build -t localhost:5000/snake-rl/inference:latest -f Inference/Dockerfile Inference/
```

### 2. Install Chart

```bash
cd ../../k8s/snake-rl

# For minikube
helm install snake-rl . -f values-minikube.yaml

# For production cluster
helm install snake-rl . -f values.yaml
```

### 3. Monitor Experiment

```bash
# Watch job status
kubectl get jobs -w

# View logs from each container
kubectl logs job/snake-rl -c clock -f    # Master coordinator
kubectl logs job/snake-rl -c env          # Game engine
kubectl logs job/snake-rl -c inference    # RL agent

# Check shared memory
kubectl exec -it job/snake-rl -c env -- df -h /dev/shm
```

## Configuration

### Experiment Settings

Edit `values.yaml` or use `--set` flags:

```bash
helm install snake-rl . \
  --set experiment.maxEpisodes=5000 \
  --set experiment.name=my-experiment
```

### Service Parameters

**Clock (Master Coordinator):**
- `clock.fps` - Game speed (0 = unlimited)
- `clock.visionSize` - Vision radius for agent

**Environment (Snake Game):**
- `env.gridWidth/gridHeight` - Game grid size
- `env.visionRadius` - Field of view radius
- `env.maxLifetime` - Max steps per episode
- `env.appleSpeed` - Apple movement speed
- `env.maxHungerSteps` - Steps before starvation
- `env.rewardConfig` - Reward structure JSON

**Inference (RL Agent):**
- `inference.learningRate` - Neural network learning rate
- `inference.gamma` - Discount factor
- `inference.beta` - Entropy regularization weight

### Resource Limits

For minikube, use `values-minikube.yaml` (lower resource requests).

For production, adjust in `values.yaml`:

```yaml
resources:
  clock:
    requests:
      cpu: 100m
      memory: 256Mi
```

### Storage

Logs and MLflow data are saved to PVC:

```yaml
storage:
  enabled: true
  size: 10Gi  # 5Gi for minikube
  storageClass: standard
```

## Accessing Results

### MLflow UI

Port-forward to access MLflow:

```bash
kubectl port-forward job/snake-rl 5000:5000
```

Then open http://localhost:5000

### Model Checkpoints

Copy models from PVC:

```bash
# List checkpoints
kubectl exec -it job/snake-rl -c clock -- ls -la /logs/*.pth

# Copy to local
kubectl cp snake-rl:/logs/model_checkpoint_episode_3000.pth ./model.pth
```

## Troubleshooting

### Job Not Starting

Check events:
```bash
kubectl describe job/snake-rl
```

### Pods Failing

Check logs:
```bash
kubectl logs job/snake-rl -c clock
kubectl logs job/snake-rl -c env
kubectl logs job/snake-rl -c inference
```

### Shared Memory Errors

Verify shared memory size:
```bash
kubectl exec -it job/snake-rl -c env -- df -h /dev/shm
```

Should show 256MB available.

### Images Not Found

Make sure images are built in minikube's Docker:
```bash
eval $(minikube docker-env)
docker images | grep snake-rl
```

## Cleanup

```bash
# Uninstall chart
helm uninstall snake-rl

# Delete PVC (optional, removes logs)
kubectl delete pvc snake-rl-logs
```

## Architecture

```
┌─────────────────────────────────────────────┐
│         Kubernetes Job: snake-rl            │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │   Container: clock (coordinator)   │    │
│  │   Container: env (game engine)     │    │
│  │   Container: inference (RL agent)  │    │
│  │                                    │    │
│  │  All run in parallel, communicate  │    │
│  │  via POSIX IPC (shared memory +    │    │
│  │  semaphores)                       │    │
│  └────────────────────────────────────┘    │
│                                             │
│  Volumes:                                   │
│  - /dev/shm (emptyDir, Memory, 256Mi)      │
│  - /logs (PVC, 10Gi)                       │
└─────────────────────────────────────────────┘
```

All containers share:
- POSIX shared memory (`/dev/shm`)
- Logs volume (`/logs`)
- IPC namespace (same Pod)

## Differences from Docker Compose

| Feature | Docker Compose | Kubernetes |
|---------|---------------|------------|
| Architecture | Separate containers | Single Pod with 3 containers |
| Shared Memory | 128MB | 256MB |
| Orchestration | `depends_on` | All containers run in parallel |
| Storage | Bind mount | PersistentVolumeClaim |
| Auto-cleanup | Manual | TTL (300s after completion) |

## Next Steps

1. Run experiment: `cd ../minikube && ./setup.sh`
2. Monitor logs: `kubectl logs job/snake-rl -c clock -f`
3. Analyze results: Use MLflow UI or copy checkpoints
4. Tune parameters: Edit `values.yaml` and reinstall
5. Scale up: Deploy to production cluster with higher resources
