# Snake RL with fog of war

## Abstract

This project is built for reinforcement learning (RL) research using a variant of the classic Snake game. The key feature is the introduction of a **fog of war** mechanism. Unlike the standard Snake, the agent (the "snake") is only able to see within a configurable field of view (FOV), a certain radius around its head. This setup is aimed at exploring partial observability in RL.


![Example how it looks Snake with Fog of War mechanic](./figures/image.png)

The full game area is not provided to the model.

Model have only the local observation as shown in the FOV.


## Current Architecture

### Service Overview

**Clock Service** (`services/Clock/src/clock.py`)
- **Role**: Master coordinator and experiment tracker
- **Functions**: Controls training loop timing, manages shared memory, orchestrates episodes via semaphores, logs metrics to MLflow
- **Key Resources**: Creates `/game_state`, `/env_control`, `/inf_control` shared memory segments

**Environment Service** (`services/Env/src/env.py`) 
- **Role**: Snake game simulation engine
- **Functions**: Runs Snake game logic, processes actions, computes rewards, writes game state to shared memory
- **Features**: Configurable grid size, vision radius, and reward structure

**Inference Service** (`services/Inference/src/inf.py`)
- **Role**: Neural network agent with integrated training
- **Functions**: Runs SnakeNet (hyperopt-optimized architecture), implements REINFORCE algorithm, maintains replay buffer
- **Training**: On-policy learning with entropy regularization

### Communication Mechanism

**Shared Memory Architecture**: Services communicate via POSIX shared memory and semaphores for maximum performance:

- **`/game_state`**: Game state (reward, game_over, action) + vision data
- **`/env_control`**: Environment control (reset commands, episode stats)  
- **`/inf_control`**: Inference control (training triggers, loss metrics)

**Synchronization Flow**:
1. Clock releases semaphores → ENV updates game state, INF selects action
2. Services signal completion → Clock coordinates next step
3. Episode end → Clock triggers training in INF service

### Development Setup

**VSCode Debug Launch** (recommended):
```
Debug (Clock + Env + Inf)  # Launches all services with debugpy
```

**Manual Start Order**:
1. Clock service (creates shared memory resources)
2. Environment service (waits for Clock resources)  
3. Inference service (waits for Clock resources)

**Cleanup**: Automatic shared memory cleanup via VSCode `clean-shm` task
