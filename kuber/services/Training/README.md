# Training Service

The Training service is the learning and model-improving component of the RL system. It receives model data, states, actions, and rewards via gRPC from the Inference service, runs a training step (using REINFORCE), and sends back updated torch JIT models.

## Purpose
- Acts as the RL "trainer" for the agent models
- Receives training batches from Inference over gRPC
- Returns improved (or new) models serialized in torch JIT format

## Key Features
- Full batch processing: supports organizing experience by episodes
- Handles cold start, model loading, error reporting to Inference
- Implements REINFORCE with entropy bonus and return normalization
- gRPC streaming for live model updates
- Logs and actions are traceable for both batch and episode granularity

## Directory Structure
```
src/
  training.py         # Main gRPC training server implementation
  requirements.txt    # Extra/overriding requirements if needed
Dockerfile
```

## Interfaces
### gRPC Service
- SendTrainingBatch: Receives a batch of states, actions, rewards, and (optional) torch JIT model from Inference, runs training, sends back TrainBatchResponse
- GetModelUpdates: Streams back model updates if Inference subscribes

Interface details are generated from protobufs in utils directory (copied in base image).

## Usage

### Build and Run (Docker)
The Dockerfile extends the base image with src code.
```Dockerfile
FROM korolaab/snake_rl_base:latest
COPY ./src /app
ENTRYPOINT []
```
Build:
```sh
docker build -t training_service .
```
Run:
```sh
docker run --network=host training_service --snake_id snake_1
```

### CLI Options
| Option          | Description                   | Default          | Required |
|-----------------|------------------------------|------------------|----------|
| --snake_id      | Snake identifier (for model)  |                  | Yes      |
| --port          | gRPC port to listen on        | 50051            | No       |
| --learning_rate | Training learning rate        | 0.001            | No       |
| --gamma         | Discount factor               | 0.99             | No       |
| --beta          | Exploration (entropy bonus)   | 0.1              | No       |
| --log_file      | Log file location             | grpc_training.log| No       |

## Training Flow (Simplified)
1. Inference service sends SendTrainingBatch with model weights and batch of (state, action, reward, done)
2. Training service processes and splits into episodes
3. Uses REINFORCE policy gradient update (returns normalized within episode)
4. Trains model and returns updated model via protobuf response

## Development Notes
- See training.py for details and customization: model can be extended or swapped
- Generated protobuf/gRPC code (training_pb2.py, training_pb2_grpc.py) must be present (generated from utils)
- Extends shared base image for torch and all dependencies
- All logs can be browsed in log_file

## Troubleshooting
- If model fails to load, Inference will receive an error
- Check for generated protobufs before running: run generate_grpc.py if needed
- All input and rewards are checked for integrity, debug logs available via log_file

---
