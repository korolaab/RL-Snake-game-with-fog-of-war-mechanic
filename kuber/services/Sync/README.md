# TCP Sync Service

A lightweight TCP synchronization service for coordinating multiple ML system components.

## Overview

This service sends TCP messages at a specified frequency to all connected clients (Inference and Environment services). Each message contains a JSON payload with a timestamp and sequence number, allowing services to synchronize their processing cycles.

The service is designed to coordinate timing between system components that need to operate in lockstep, ensuring that inference, environment updates, and other operations happen in a synchronized manner.

## Key Features

- Configurable frequency (default: 10Hz)
- TCP server, multiple clients can connect (Inference and Environment)
- JSON-formatted sync signals with timestamps
- Maintains consistent timing intervals
- Minimal dependencies (standard Python libraries only)

## Command-Line Arguments

The sync service accepts the following command-line arguments:

