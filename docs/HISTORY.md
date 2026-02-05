# Evolution

## 1. Monolithic (legacy/)
Single process, Pygame visualization. Hard to modify individual components.

Results: [legacy/README.md](../legacy/README.md)

## 2. REST/gRPC (abandoned)
Attempted microservices with Flask REST API and gRPC.

**Why failed:** Network latency too high (10-50ms/step), training 10x slower.

**Commits:** 552a84a (Flask), f50c3e4 (gRPC), b9a2aea (UDP)

## 3. Shared Memory (current)
POSIX IPC via `/dev/shm`. Three services: Clock, Env, Inference.

**Advantages:** Near-zero latency, modular, easy to modify components.

**Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
