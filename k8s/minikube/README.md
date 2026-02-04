# Snake RL on Minikube

Run Snake RL experiments locally using minikube.

## Prerequisites

Install required tools:

### 1. Minikube

```bash
# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# macOS
brew install minikube

# Verify
minikube version
```

### 2. kubectl

```bash
# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/kubectl

# macOS
brew install kubectl

# Verify
kubectl version --client
```

### 3. Helm

```bash
# Linux/macOS
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify
helm version
```

## Quick Start (Automated)

Run the setup script to automatically:
- Start minikube
- Build Docker images
- Install Helm chart
- Start experiment

```bash
./setup.sh
```

That's it! The experiment will run for 3000 episodes.

## Quick Start (Manual)

### 1. Start Minikube

```bash
minikube start --cpus=4 --memory=8192
```

### 2. Build Images

Configure Docker to use minikube's daemon:

```bash
eval $(minikube docker-env)
```

Build images:

```bash
cd ../../services

# Base image (5-10 minutes)
docker build -t korolaab/snake_rl_base:latest -f Dockerfile .

# Service images (30 seconds each)
docker build -t localhost:5000/snake-rl/clock:latest -f Clock/Dockerfile Clock/
docker build -t localhost:5000/snake-rl/env:latest -f Env/Dockerfile Env/
docker build -t localhost:5000/snake-rl/inference:latest -f Inference/Dockerfile Inference/
```

### 3. Install Chart

```bash
cd ../k8s/snake-rl
helm install snake-rl . -f values-minikube.yaml
```

### 4. Monitor

```bash
# Watch job status
kubectl get jobs -w

# View logs
kubectl logs job/snake-rl -c clock -f
```

## Common Commands

### Check Status

```bash
# Minikube status
minikube status

# Kubernetes resources
kubectl get all

# Job status
kubectl get jobs

# Pod logs
kubectl logs job/snake-rl -c clock    # Master coordinator
kubectl logs job/snake-rl -c env      # Game engine
kubectl logs job/snake-rl -c inference # RL agent
```

### Access Results

```bash
# List generated files
kubectl exec -it job/snake-rl -c clock -- ls -la /logs

# Copy model checkpoint
kubectl cp snake-rl:/logs/model_checkpoint_episode_3000.pth ./model.pth

# Port-forward MLflow (if UI is available)
kubectl port-forward job/snake-rl 5000:5000
```

### Cleanup

```bash
# Uninstall chart
helm uninstall snake-rl

# Delete PVC (optional, removes logs)
kubectl delete pvc snake-rl-logs

# Stop minikube
minikube stop

# Delete minikube cluster
minikube delete
```

## Troubleshooting

### Minikube Won't Start

```bash
# Check system resources
free -h  # Need 8GB+ RAM available

# Try with fewer resources
minikube start --cpus=2 --memory=4096

# Check logs
minikube logs
```

### Images Not Found

Make sure you're using minikube's Docker:

```bash
eval $(minikube docker-env)
docker images | grep snake-rl
```

If images are missing, rebuild them.

### Job Fails Immediately

Check events:

```bash
kubectl describe job/snake-rl
kubectl describe pod -l app.kubernetes.io/name=snake-rl
```

Common issues:
- Image pull errors → Use `imagePullPolicy: Never`
- Resource limits → Reduce in `values-minikube.yaml`
- Shared memory size → Check `/dev/shm` mount

### Shared Memory Errors

Verify shared memory is mounted:

```bash
kubectl exec -it job/snake-rl -c env -- df -h /dev/shm
```

Should show 256MB available.

### PVC Won't Bind

Check storage class:

```bash
kubectl get storageclass

# Minikube should have 'standard' as default
```

If not, create one or use emptyDir by setting `storage.enabled: false`.

## Configuration

### Resource Limits

For lower-spec machines, reduce resources in `values-minikube.yaml`:

```yaml
resources:
  clock:
    requests:
      cpu: 50m
      memory: 128Mi
  # ... etc
```

### Experiment Parameters

Override values during install:

```bash
helm install snake-rl ../snake-rl \
  -f ../snake-rl/values-minikube.yaml \
  --set experiment.maxEpisodes=1000 \
  --set env.gridWidth=7 \
  --set env.gridHeight=7
```

## Performance Tips

1. **CPU allocation**: Give minikube 4+ CPUs for faster training
2. **Memory**: 8GB minimum, 16GB recommended
3. **Docker cache**: First build is slow (~10 min), rebuilds are fast
4. **Shared memory**: 256MB is enough for most experiments

## Architecture in Minikube

```
┌──────────────────────────────────────────┐
│          Minikube Cluster                │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │    Kubernetes Job: snake-rl        │ │
│  │                                    │ │
│  │  Init: clock → Containers: env,   │ │
│  │                           inference│ │
│  │                                    │ │
│  │  Shared Memory: /dev/shm (256Mi)  │ │
│  │  PVC: snake-rl-logs (5Gi)         │ │
│  └────────────────────────────────────┘ │
│                                          │
│  Storage: HostPath (minikube VM)        │
└──────────────────────────────────────────┘
```

## Next Steps

1. Run automated setup: `./setup.sh`
2. Watch training: `kubectl logs job/snake-rl -c clock -f`
3. Experiment with parameters in `../snake-rl/values-minikube.yaml`
4. Scale up to production cluster when ready

## Resources

- [Minikube Docs](https://minikube.sigs.k8s.io/docs/)
- [Kubernetes Docs](https://kubernetes.io/docs/)
- [Helm Docs](https://helm.sh/docs/)
- Project README: `../snake-rl/README.md`
