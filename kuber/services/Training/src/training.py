import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import logging
import os
import sys
import glob
import pickle
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional


def setup_logger(log_file: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def find_latest_files(folder_path: str) -> Dict[str, Optional[str]]:
    """Find the latest model, history and weights files in the folder."""
    files = {
        'model': None,
        'history': None,
        'weights': None
    }
    
    # Find latest model file
    model_pattern = os.path.join(folder_path, "snake_model_*.pth")
    model_files = glob.glob(model_pattern)
    if model_files:
        model_files.sort(key=lambda x: os.path.getctime(x))
        files['model'] = model_files[-1]
        logging.info(f"Found latest model: {files['model']}")
    
    # Find latest history file
    history_pattern = os.path.join(folder_path, "history_*.pkl")
    history_files = glob.glob(history_pattern)
    if history_files:
        history_files.sort(key=lambda x: os.path.getctime(x))
        files['history'] = history_files[-1]
        logging.info(f"Found latest history: {files['history']}")
    
    # Find latest weights file
    weights_pattern = os.path.join(folder_path, "weights_*.pkl")
    weights_files = glob.glob(weights_pattern)
    if weights_files:
        weights_files.sort(key=lambda x: os.path.getctime(x))
        files['weights'] = weights_files[-1]
        logging.info(f"Found latest weights: {files['weights']}")
    
    return files


def load_model_from_file(model_path: str) -> Tuple[nn.Module, Dict]:
    """Load model architecture from saved file."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_architecture' not in checkpoint:
            raise ValueError("Model architecture not found in checkpoint")
        
        model = checkpoint['model_architecture']
        model_info = {
            'snake_id': checkpoint.get('snake_id', 'unknown'),
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'model_path': model_path
        }
        
        logging.info(f"Loaded model architecture from {model_path}")
        logging.info(f"Model info: {model_info}")
        
        return model, model_info
        
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise


def load_history_from_file(history_path: str) -> List[Dict]:
    """Load training history from file."""
    try:
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        logging.info(f"Loaded history with {len(history)} experiences from {history_path}")
        return history
        
    except Exception as e:
        logging.error(f"Error loading history from {history_path}: {e}")
        raise


def load_weights_from_file(weights_path: str) -> Dict:
    """Load model weights from file."""
    try:
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
        
        logging.info(f"Loaded weights from {weights_path}")
        return weights
        
    except Exception as e:
        logging.error(f"Error loading weights from {weights_path}: {e}")
        raise


def apply_weights_to_model(model: nn.Module, weights: Dict):
    """Apply loaded weights to the model."""
    try:
        # Convert numpy arrays back to tensors and load into model
        state_dict = {}
        for name, weight in weights.items():
            if isinstance(weight, np.ndarray):
                state_dict[name] = torch.tensor(weight)
            else:
                state_dict[name] = weight
        
        model.load_state_dict(state_dict)
        logging.info("Successfully applied weights to model")
        
    except Exception as e:
        logging.error(f"Error applying weights to model: {e}")
        raise


def process_history_for_training(history: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process history data for training.
    Extract states and rewards, convert to tensors.
    
    This is a simple approach - you might want to modify this
    based on your specific training strategy.
    """
    states = []
    rewards = []
    
    for experience in history:
        state = experience['state']
        reward = experience['reward']
        
        # Extract visible cells and process similar to state_processor
        visible_cells = state.get('visible_cells', {})
        filtered_cells = {k: v for k, v in visible_cells.items() if v != 'HEAD'}
        
        if not filtered_cells:
            continue
            
        # Sort cells by coordinates
        sorted_cells = []
        for coord_str, cell_type in filtered_cells.items():
            try:
                x, y = map(int, coord_str.split(','))
                sorted_cells.append((x, y, cell_type))
            except ValueError:
                continue
        
        if not sorted_cells:
            continue
            
        sorted_cells.sort(key=lambda item: (item[0], item[1]))
        
        # Encode cells
        cell_encoding = {
            'FOOD': [0, 0, 1],
            'BODY': [0, 1, 0],
            'OTHER_BODY': [1, 0, 1],
            'EMPTY': [0, 0, 0]
        }
        
        tensor_data = []
        for x, y, cell_type in sorted_cells:
            encoding = cell_encoding.get(cell_type, [0, 0, 0])
            tensor_data.append(encoding)
        
        if tensor_data:
            # Flatten the tensor for model input
            state_tensor = torch.tensor(tensor_data, dtype=torch.float32).flatten()
            states.append(state_tensor)
            rewards.append(reward)
    
    if not states:
        raise ValueError("No valid states found in history")
    
    # Pad states to same length (take maximum length)
    max_length = max(len(state) for state in states)
    padded_states = []
    
    for state in states:
        if len(state) < max_length:
            # Pad with zeros
            padded_state = torch.zeros(max_length)
            padded_state[:len(state)] = state
            padded_states.append(padded_state)
        else:
            padded_states.append(state)
    
    states_tensor = torch.stack(padded_states)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    logging.info(f"Processed {len(states)} states, tensor shape: {states_tensor.shape}")
    
    return states_tensor, rewards_tensor


def perform_training_step(model: nn.Module, states: torch.Tensor, rewards: torch.Tensor, 
                         learning_rate: float = 0.001) -> Dict[str, float]:
    """
    Perform one training step.
    
    This is a simple supervised learning approach where we try to predict
    rewards from states. You might want to implement a more sophisticated
    training strategy like reinforcement learning.
    """
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Forward pass
    model.train()
    predictions = model(states)
    
    # Simple loss: try to predict if reward is positive
    # Convert rewards to classification targets (positive reward = class 1, else class 0)
    targets = (rewards > 0).long()
    
    # Use cross-entropy loss for classification
    loss = F.cross_entropy(predictions, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    with torch.no_grad():
        predicted_classes = torch.argmax(predictions, dim=1)
        accuracy = (predicted_classes == targets).float().mean()
    
    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'num_samples': len(states),
        'positive_rewards': (rewards > 0).sum().item(),
        'negative_rewards': (rewards <= 0).sum().item()
    }
    
    return metrics


def save_updated_model(model: nn.Module, folder_path: str, original_info: Dict, 
                      metrics: Dict[str, float]) -> str:
    """Save the updated model after training."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"trained_model_{original_info.get('snake_id', 'unknown')}_{timestamp}.pth"
    model_path = os.path.join(folder_path, model_filename)
    
    # Create optimizer for saving (we don't need to preserve its state for inference)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': model,
        'optimizer_state_dict': optimizer.state_dict(),
        'snake_id': original_info.get('snake_id', 'unknown'),
        'timestamp': timestamp,
        'training_metrics': metrics,
        'trained_from': original_info.get('model_path', 'unknown')
    }
    
    torch.save(checkpoint, model_path)
    logging.info(f"Saved updated model to: {model_path}")
    
    return model_path


def save_training_log(folder_path: str, metrics: Dict[str, float], 
                     original_info: Dict) -> str:
    """Save training metrics to a JSON log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.json"
    log_path = os.path.join(folder_path, log_filename)
    
    log_data = {
        'timestamp': timestamp,
        'original_model_info': original_info,
        'training_metrics': metrics
    }
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    logging.info(f"Saved training log to: {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(description="Snake Agent Training Script")
    parser.add_argument("--folder", required=True, help="Folder containing model, history, and weights files")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--log_file", default="training.log", help="Path to the log file")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(args.log_file)
    
    logging.info(f"Starting training with folder: {args.folder}")
    logging.info(f"Learning rate: {args.learning_rate}")
    
    try:
        # Find latest files
        files = find_latest_files(args.folder)
        
        if not files['model']:
            raise ValueError("No model file found in folder")
        if not files['history']:
            raise ValueError("No history file found in folder")
        if not files['weights']:
            raise ValueError("No weights file found in folder")
        
        # Load model architecture
        model, model_info = load_model_from_file(files['model'])
        
        # Load and apply weights
        weights = load_weights_from_file(files['weights'])
        apply_weights_to_model(model, weights)
        
        # Load history
        history = load_history_from_file(files['history'])
        
        # Process history for training
        states, rewards = process_history_for_training(history)
        
        # Perform training step
        logging.info("Performing training step...")
        metrics = perform_training_step(model, states, rewards, args.learning_rate)
        
        # Log results
        logging.info("Training step completed!")
        logging.info(f"Loss: {metrics['loss']:.4f}")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Samples: {metrics['num_samples']}")
        logging.info(f"Positive rewards: {metrics['positive_rewards']}")
        logging.info(f"Negative rewards: {metrics['negative_rewards']}")
        
        # Save updated model
        updated_model_path = save_updated_model(model, args.folder, model_info, metrics)
        
        # Save training log
        log_path = save_training_log(args.folder, metrics, model_info)
        
        logging.info("Training completed successfully!")
        logging.info(f"Updated model saved to: {updated_model_path}")
        logging.info(f"Training log saved to: {log_path}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
