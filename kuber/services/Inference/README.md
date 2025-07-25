# Inference Service

The Inference service provides the real-time control block of your RL learning setup. It connects to the environment via REST API, receives a stream of state updates, and returns actions (commands) to the environment for agent control. This service also interacts with the Training component using gRPC for model updates.

## Purpose

- Acts as the decision-maker in the RL pipeline
- Receives live game state from the environment over REST
- Sends back move decisions (actions) via REST
- Supports model update and training interaction through gRPC

## Key Features
- Asynchronous state stream reading and action sending
- Can work in async mode or (optionally) use **TCP Sync** (with "Sync" service)
- Designed for integration with a distributed RL pipeline (inference, training, environment)

## Directory Structure

