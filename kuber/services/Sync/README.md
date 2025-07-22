# UDP Sync Service

A lightweight UDP synchronization service for coordinating multiple ML system components.

## Overview

This service sends UDP packets at a specified frequency to both the Inference and Environment services. Each packet contains a JSON payload with a timestamp and sequence number, allowing services to synchronize their processing cycles.

The service is designed to coordinate timing between system components that need to operate in lockstep, ensuring that inference, environment updates, and other operations happen in a synchronized manner.

## Key Features

- Configurable frequency (default: 10Hz)
- Separate host/port configuration for Inference and Environment services
- JSON-formatted sync signals with timestamps
- Maintains consistent timing intervals
- Minimal dependencies (standard Python libraries only)

## Command-Line Arguments

The sync service accepts the following command-line arguments:

```
--inference-host     Hostname/IP of the Inference service (default: localhost)
--inference-port     UDP port for the Inference service (default: 5555)
--env-host           Hostname/IP of the Environment service (default: localhost)
--env-port           UDP port for the Environment service (default: 5555)
--frequency          Frequency to send sync signals in Hz (default: 10.0)
--verbose            Enable verbose logging
```

## Example Usage

Basic usage with default settings (sends sync signals at 10Hz to both services on port 5555):

```bash
python main.py
```

Custom configuration:

```bash
python main.py --inference-host=inference-service --inference-port=5555 --env-host=env-service --env-port=5556 --frequency=20 --verbose
```

## How It Works

1. The service creates a UDP socket
2. It enters a loop that:
   - Creates a timestamp
   - Sends a UDP packet to both services
   - Sleeps to maintain the specified frequency
3. The service continues until interrupted (e.g., Ctrl+C)

## Service Integration

The receiving services (Inference and Environment) should be configured to listen on the specified ports with the `sync_enabled=True` and appropriate `sync_port` parameters.

### Inference Service Configuration:

```bash
python main.py --sync_enabled --sync_port=5555 ...
```

### Environment Service Configuration:

```bash
python main.py --sync_enabled --sync_port=5555 ...
```

## Signal Format

The synchronization signal is a JSON-encoded packet with the following structure:

```json
{
  "timestamp": "2023-07-15T12:34:56.789012",
  "sequence": 42
}
```

- `timestamp`: ISO 8601 formatted timestamp
- `sequence`: Incrementing sequence number starting from 1