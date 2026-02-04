# Quick Start Guide - Docker Compose

Run Snake RL experiments using Docker Compose with shared memory architecture.

## Prerequisites

- Docker and Docker Compose
- 8GB+ RAM available
- 10GB+ free disk space

## Quick Start (5 minutes)

### 1. Build Images

```bash
# Build base image (5-10 minutes, one-time)
docker build -t korolaab/snake_rl_base:latest -f Dockerfile .

# Build service images (30 seconds each)
docker build -t localhost:5000/snake-rl/clock:latest -f Clock/Dockerfile Clock/
docker build -t localhost:5000/snake-rl/env:latest -f Env/Dockerfile Env/
docker build -t localhost:5000/snake-rl/inference:latest -f Inference/Dockerfile Inference/
```

### 2. Run Experiment

```bash
docker compose up
```

Watch it train! The experiment runs 3000 episodes with MLflow tracking.

### 3. View Results

```bash
# After experiment completes
cd logs-storage/
mlflow ui --backend-store-uri file:///logs/mlruns
```

Open http://localhost:5000 to explore metrics.

## Configuration

Edit `docker-compose.yaml` to change:

- **Episodes**: `--max-episodes 3000`
- **Grid size**: `--grid_width 11 --grid_height 11`
- **Learning rate**: `--learning-rate 0.001`
- **Apple behavior**: `--apple-speed 0.5 --max-hunger-steps 150`

## Troubleshooting

**"connection refused" error:**
```bash
# Add pull_policy: never to each service in docker-compose.yaml
```

**Out of memory:**
```bash
# Reduce batch size or grid size in docker-compose.yaml
```

**Shared memory errors:**
```bash
# Increase shared memory volume size in docker-compose.yaml
o: size=256m  # Change from 128m to 256m
```

## Architecture

```
┌──────────────────────────────────────────────┐
│          Shared Memory (/dev/shm)            │
│  /game_state  /env_control  /inf_control     │
└──────────────────────────────────────────────┘
        ↑              ↑              ↑
        │              │              │
   ┌────┴────┐    ┌───┴────┐    ┌────┴─────┐
   │  Clock  │───→│  Env   │───→│Inference │
   │Coordin. │    │ Snake  │    │ RL Agent │
   │ MLflow  │    │ Game   │    │ Training │
   └─────────┘    └────────┘    └──────────┘
```

## Output Files

- `logs-storage/*.pth` - Model checkpoints
- `logs-storage/mlruns/` - MLflow experiment data
- Container logs - Training metrics per episode

## Next Steps

1. **Tune hyperparameters** - Edit learning rate, gamma, entropy weight
2. **Change environment** - Adjust grid size, apple speed, hunger mechanics
3. **Analyze results** - Use MLflow UI to compare experiments
4. **Export models** - Copy `.pth` files for deployment

For more details, see main [README.md](../README.md).
