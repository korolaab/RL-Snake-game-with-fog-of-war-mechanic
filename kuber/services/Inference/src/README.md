# Neural Snake Agent

A PyTorch-based neural network agent for playing snake games. The agent uses a feed-forward neural network to make decisions based on the game state and automatically saves/loads models between sessions.


## Features

- **Neural Network Decision Making**: Uses a 3-layer neural network with Tanh activation and dropout
- **Automatic Model Management**: Automatically loads the latest saved model on startup or creates a new one
- **State Processing**: Converts game state into tensor format suitable for neural network input
- **Experience Tracking**: Saves game history (states and rewards) for analysis
- **Modular Architecture**: Clean separation of concerns across multiple modules

## Project Structure

```
├── main.py              # Main entry point and game loop
├── snake_agent.py       # Main agent class that orchestrates everything
├── snake_model.py       # Neural network architecture and model management
├── state_processor.py   # Game state to tensor conversion
├── data_manager.py      # Data saving/loading and experience management
└── models/              # Directory for saved models and data
```

## Installation

### Requirements

```bash
pip install torch requests numpy
```

### Dependencies

- Python 3.7+
- PyTorch
- requests
- numpy

## Usage

### Basic Usage

```bash
python main.py --snake_id snake_1 --env_host localhost:8000
```

### Advanced Usage

```bash
python main.py \
    --snake_id snake_1 \
    --env_host localhost:8000 \
    --model_save_dir models \
    --learning_rate 0.001 \
    --log_file agent.log
```

### Command Line Arguments

- `--snake_id`: Unique identifier for the snake (required)
- `--env_host`: Host address of the snake game server (required)
- `--model_save_dir`: Directory to save models and data (default: "models")
- `--learning_rate`: Learning rate for the optimizer (default: 0.001)
- `--log_file`: Path to the log file (default: "neural_agent.log")

## How It Works

### Game State Processing

The agent receives game states containing visible cells in a diamond-shaped field of view. Each cell can be:

- `HEAD`: The snake's head (ignored in processing)
- `BODY`: The snake's own body
- `OTHER_BODY`: Other snakes' bodies
- `FOOD`: Food items
- `EMPTY`: Empty cells

### State Encoding

Each cell type is encoded as a 3-element vector:

- `FOOD`: [0, 0, 1]
- `BODY`: [0, 1, 0]
- `OTHER_BODY`: [1, 0, 1]
- `EMPTY`: [0, 0, 0]

The visible cells are sorted by coordinates (x, y) and concatenated into a flat tensor for neural network input.

### Neural Network Architecture

```python
SnakeNet(
    Linear(input_size, 64),
    Tanh(),
    Dropout(0.5),
    Linear(64, 16),
    Tanh(),
    Dropout(0.5),
    Linear(16, 3)  # 3 actions: left, right, forward
)
```

### Actions

The agent can choose from three actions:
- `left`: Turn left
- `right`: Turn right  
- `forward`: Continue straight (no action sent)

## Model Management

### Automatic Model Loading

On startup, the agent:
1. Searches for existing models for the current `snake_id`
2. If none found, searches for any available models
3. Loads the most recent model by file creation time
4. If no models exist, creates a new one on first game state

### Model Saving

When the game ends (`game_over = true`), the agent saves:
1. **Model file** (.pth): Complete model with architecture, weights, and optimizer state
2. **History file** (.pkl): All game states and rewards from the session
3. **Weights file** (.pkl): Model weights in numpy format for analysis

### File Naming Convention

```
snake_model_{snake_id}_{timestamp}.pth
history_{snake_id}_{timestamp}.pkl
weights_{snake_id}_{timestamp}.pkl
```

## Module Details

### snake_model.py

- `SnakeNet`: Neural network architecture
- `ModelManager`: Handles model creation, loading, and saving operations

### state_processor.py

- `StateProcessor`: Converts game states to tensors
- Handles coordinate parsing and cell type encoding

### data_manager.py

- `DataManager`: Manages experience history and data persistence
- Provides statistics on collected data

### snake_agent.py

- `NeuralSnakeAgent`: Main agent class that coordinates all components
- Handles model initialization, action prediction, and data collection

### main.py

- Network communication with game server
- Main game loop and error handling
- Command line interface

## Loading Saved Models

To load a saved model in another script:

```python
from main import load_saved_model

# Load a specific model
model, optimizer, info = load_saved_model("models/snake_model_snake_1_20250605_123456.pth")

print(f"Loaded model trained for: {info['snake_id']}")
print(f"Model timestamp: {info['timestamp']}")
```

## Game Server Requirements

The agent expects a game server that provides:

### State Stream Endpoint
`GET http://{env_host}/snake/{snake_id}`

Returns JSON lines with game states:
```json
{
    "snake_id": "snake_1",
    "visible_cells": {
        "5,5": "HEAD",
        "4,5": "BODY",
        "6,5": "FOOD",
        "7,5": "EMPTY"
    },
    "reward": 1,
    "game_over": false
}
```

### Move Endpoint
`POST http://{env_host}/snake/{snake_id}/move`

Accepts JSON payload:
```json
{
    "move": "left"  // or "right"
}
```

## Logging

The agent provides detailed logging including:
- Model loading/creation status
- Action predictions and probabilities
- Game statistics and performance metrics
- Error handling and debugging information

Log levels can be controlled through the logging configuration in `setup_logger()`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Feel free to use and modify as needed.

## Troubleshooting

### Common Issues

**Model size mismatch**: If the game field size changes, the agent will automatically recreate the model with the correct input size.

**Connection errors**: Ensure the game server is running and accessible at the specified host.

**Permission errors**: Make sure the agent has write permissions to the model save directory.

### Debug Mode

For more detailed logging, modify the logging level in `setup_logger()`:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```
