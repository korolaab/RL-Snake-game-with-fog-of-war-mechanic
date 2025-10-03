# Clock Service

Sequential orchestrator for ENV and INF services in the Snake RL system.

## Overview

The Clock service ensures deterministic sequential execution by controlling when ENV and INF services are allowed to run. It uses POSIX shared memory and semaphores for high-performance inter-process communication.

## Architecture

```
Clock Service Cycle:
1. Release sem_env_trigger → ENV executes one step
2. Wait for sem_env_ready   → ENV signals completion  
3. Release sem_inf_trigger → INF executes one step
4. Wait for sem_inf_ready  → INF signals completion
5. Repeat cycle
```

## Dependencies

- `posix_ipc` - POSIX shared memory and semaphore support
- Python 3.11+

## Configuration

### Command Line Arguments

- `--shm-size`: Shared memory size in bytes (default: 1024)
- `--tick-interval`: Minimum time between cycles in seconds (default: 0.001)
- `--log-level`: Logging level DEBUG/INFO/WARNING/ERROR (default: INFO)

### Example Usage

```bash
# Basic usage
python main.py

# Custom configuration
python main.py --shm-size 2048 --tick-interval 0.01 --log-level DEBUG
```

## Docker Usage

```bash
# Build image
docker build -t snake-rl/clock .

# Run container
docker run --rm \
  -v /dev/shm:/dev/shm \
  snake-rl/clock
```

## Shared Memory Layout

The Clock service creates a shared memory block at `/snake_game_state` that ENV and INF services use to exchange state and action data.

## Semaphores

- `/sem_env_trigger` - Clock signals ENV to execute
- `/sem_env_ready` - ENV signals Clock when step complete  
- `/sem_inf_trigger` - Clock signals INF to execute
- `/sem_inf_ready` - INF signals Clock when step complete

## Error Handling

- Service timeouts: 5 second timeout for ENV/INF step completion
- Graceful shutdown: Handles SIGINT/SIGTERM signals
- Resource cleanup: Automatically cleans shared memory and semaphores on exit
- Service detection: Waits for ENV/INF services to connect before starting

## Monitoring

The service logs cycle milestones every 1000 cycles and provides detailed debug logging of the sequential execution flow.