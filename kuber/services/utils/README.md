# Utils Directory

The utils directory contains shared utilities and protocol definitions for all services in the RL system. These provide shared logging, protobuf-based interface definitions, and support scripts relevant to service development and deployment.

## Contents

- `logger.py` — Flexible, JSON-based, multi-backend logger for all RL services. Supports output to console, file, and RabbitMQ. Used by all services for experiment tracking, error tracing, and debugging.
- `training.proto` — Protocol buffer (protobuf) definition for the training gRPC API. Used to generate the Python files for communications between Inference and Training services.
- `generate_grpc.py` — Script to generate Python gRPC/protobuf files from definitions (run before using Training).
- `setup_protobuf.sh` — Shell script to install/generate protobuf dependencies if needed
- `rabbitmq_handler.py` — (Optional) Dedicated handler for logging/output to RabbitMQ, used by logger

## How to Use

### Logging (logger.py)
- All services should call the logger's setup function (usually via `logger.setup_as_default()`) on startup.
- Supports INFO, DEBUG, WARNING, ERROR, and CRITICAL with metadata (container, experiment, run ID)
- Can be configured using env vars, CLI args, or from Python

Example:
```python
import logger
logger.setup_as_default(experiment_name="exp1", log_file="experiment.log")

import logging
logging.info({"event": "start", "service": "training"})
```

### Protobuf Setup
If you update `training.proto`, re-run the script:
```bash
python generate_grpc.py
```
This will generate `training_pb2.py` and `training_pb2_grpc.py` for use in Training and Inference services. (These are copied into the snake_rl_base image for all services.)

### Dependencies
- All required Python dependencies are handled by the base image.
- For development: install grpcio, grpcio-tools, protobuf, and optionally pika for RabbitMQ support.

### Best Practices
- Treat this folder as the only source of truth for message formats (protobuf) and logging.
- Always version or lock the base image if you update or extend these utilities.

---
