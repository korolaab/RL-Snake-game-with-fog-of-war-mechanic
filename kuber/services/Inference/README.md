# Inference Service

The Inference service provides the real-time control block of your RL learning setup. It connects to the environment via REST API, receives a stream of state updates, and returns actions (commands) to the environment for agent control. This service also interacts with the Training component using gRPC for model updates.

## Purpose

- Acts as the decision-maker in the RL pipeline
- Receives live game state from the environment over REST
- Sends back move decisions (actions) via REST
- Supports model update and training interaction through gRPC

## Key Features
- Asynchronous state stream reading and action sending
- Can work in async mode or (optionally) use UDP Sync (with "Sync" service, under development)
- Designed for integration with a distributed RL pipeline (inference, training, environment)

## Directory Structure

```
src/
  main.py             # Main inference loop and REST/gRPC logic
  snake_agent.py      # Agent logic and experience management
  snake_model.py      # Model architecture and serialization
  state_processor.py  # State encoding logic
  data_manager.py     # Experience, data, and batch-handling
  random_infer.py     # (likely alternative or test agent)
Dockerfile
```

## Interfaces

### REST API (Environment communication)
- GET {env_host}/snake/{snake_id}: stream of JSON states
- POST {env_host}/snake/{snake_id}/move: send back action (`{"move": "left"/"right"}`)

### gRPC (Training communication)
- Exchanges full model (weights + architecture) and episode batches for training

### UDP Sync (optional, for coordinated step-by-step env/agent interaction)
- Activated if Sync service is deployed; otherwise, operates in async mode

## Usage

### Build and Run (Docker)
Each Inference service has its own Dockerfile, extending the base image:
```Dockerfile
FROM korolaab/snake_rl_base:latest
COPY ./src /app
ENTRYPOINT []
```

Build:
```sh
docker build -t inference_service .
```
Run:
```sh
docker run --network=host inference_service --snake_id snake_1 --env_host localhost:8000
```

### CLI Options

| Option                | Description                              | Default      | Required |
|-----------------------|------------------------------------------|--------------|----------|
| --snake_id            | Unique ID for this snake agent           |              | Yes      |
| --env_host            | Env REST server host:port                |              | Yes      |
| --model_save_dir      | Directory to save/load models            | models/      | No       |
| --learning_rate       | Learning rate for training               | 0.001        | No       |
| --grpc_host           | Training gRPC server host                | localhost    | No       |
| --grpc_port           | Training gRPC server port                | 50051        | No       |
| --batch_size          | Episodes per training batch (gRPC batch) | 5            | No       |
| --sync_enabled        | Enable UDP sync with Sync service        | false        | No       |
| --sync_port           | UDP port for sync listen                 | 5555         | No       |
| --log_file            | Path for logging                         | neural_agent.log | No   |

## High-Level Pipeline

1. Connect to Env stream and receive state updates (JSON)
2. Encode state & forward to the agent/model for action prediction
3. Send action back to environment via REST
4. Collect episode data into batches (size = batch_size)
5. Communicate with Training service via gRPC for model updates
6. Optionally react to Sync service signals for step-based evaluation

## Model Management

- Loads/reuses existing torch JIT models (*.pth)
- Stores new models and training history after episodes or batch training

## Deployment Notes

- The Dockerfile expects to run with the "snake_rl_base" image, which must include all Python, torch, and dependency requirements for all RL services.
- Each service can be deployed and scaled independently (multiple inference agents per environment/snakes supported).

## Troubleshooting

- Ensure the environment server and training server are accessible from the container.
- Use log_file for debugging.
- If Sync service is not running, run without --sync_enabled.

## Development

- The agent, model, and state-processing logic is Pythonic and self-contained: see main.py and related src files for customization/extension.
- Use random_infer.py, snake_agent.py, and others for additional agent implementations and debugging.
