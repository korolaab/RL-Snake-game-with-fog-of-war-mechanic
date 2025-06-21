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
import onnx
import onnx2torch
import traceback

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

def load_model(latest_model_path, state_model_path, learning_rate: float = 0.001): 
    """Load the latest ONNX model for the snake and convert to PyTorch."""                           
                                                                            
    if latest_model_path is None:                                           
        logging.info("No existing ONNX models found.")                           
        return None, None, None                                             
                                                                            
    return load_onnx_model(latest_model_path, state_model_path, learning_rate)

def load_onnx_model(model_path: str, state_model_path: str, learning_rate: float = 0.001):
    """Load ONNX model and convert to PyTorch model."""
    try:
        # Load ONNX model
        onnx_model = onnx.load(model_path)
        
        # Convert ONNX to PyTorch
        pytorch_model = onnx2torch.convert(onnx_model)
        
        # Set model to training mode
        pytorch_model.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=learning_rate)
        
        # You might want to load additional training state if available
        # Check for accompanying state file (e.g., .pth file with same name)
        epoch = 0
        
        if os.path.exists(state_model_path):
            try:
                checkpoint = torch.load(state_model_path, map_location='cpu')
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epoch' in checkpoint:
                    epoch = checkpoint['epoch']
                logging.info(f"Loaded training state from {state_path}")
            except Exception as e:
                logging.warning(f"Could not load training state: {e}")
        
        logging.info(f"Successfully loaded ONNX model from {model_path}")
        
        return pytorch_model, optimizer, epoch
        
    except Exception as e:
        logging.error(f"Failed to load ONNX model from {model_path}: {e}")
        return None, None, None 

def save_model(model, input_size: int, optimizer, snake_id: str, epoch:int, model_save_dir: str, additional_data: dict = None):
    """Save PyTorch model as ONNX format with additional training state."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #TODO epoch in filenames
    
    # ONNX model path
    onnx_model_path = os.path.join(model_save_dir, f"snake_model_{snake_id}_{timestamp}.onnx")
    
    # Training state path (for optimizer and additional data)
    state_path = os.path.join(model_save_dir, f"snake_model_state_{snake_id}_{timestamp}.pth")
    
    try:
        # Set model to evaluation mode for export
        model.eval()
        
        # Create dummy input for tracing (adjust dimensions based on your model)
        # You'll need to modify this based on your actual input shape
        dummy_input = torch.randn(1, input_size)  # Example: batch_size=1, input_features=4
        
        # Export model to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Save training state separately
        checkpoint = {
            'optimizer_state_dict': optimizer.state_dict(),
            'snake_id': snake_id,
            'timestamp': timestamp,
            'model_input_shape': list(dummy_input.shape),  # Store input shape info
        }
        
        # Add additional data if provided
        if additional_data:
            checkpoint.update(additional_data)
        
        torch.save(checkpoint, state_path)
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        
        logging.info(f"ONNX model saved to: {onnx_model_path}")
        logging.info(f"Training state saved to: {state_path}")
        
        # Set model back to training mode
        model.train()
        
        return onnx_model_path, state_path
    
    except Exception as e:
        logging.error(f"Failed to save model as ONNX: {e}")
        return None, None


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

def process_history_for_training(history: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process history data for training.
    Extract states and rewards, convert to tensors.
    
    This is a simple approach - you might want to modify this
    based on your specific training strategy.
    """
    states = []
    rewards = []
    actions = []
    
    for experience in history:
        state = experience['state']
        reward = experience['reward']
        action = experience['action']
        
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
            'OTHER_BODY': [1, 0, 0],
            'EMPTY': [0, 0, 0]
        }
        
        tensor_cell_data = []
        for x, y, cell_type in sorted_cells:
            encoding = cell_encoding.get(cell_type, [0, 0, 0])
            tensor_cell_data.append(encoding)

        # Encode actions
        action_encoding = {
                'forward': [0,0,1],
                'right': [0,1,0],
                'left':[1,0,0]
                }
        
        action_vector = action_encoding.get(action, [0, 0, 1])

        rewards.append(reward)

        state_tensor = torch.tensor(tensor_cell_data, dtype=torch.float32).flatten()
        states.append(state_tensor)

        action_tensor = torch.tensor(action_vector, dtype=torch.float32)
        actions.append(action_tensor)

    
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
    actions_tensor = torch.stack(actions)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    logging.info(f"Processed {len(states)} states, tensor shape: {states_tensor.shape}")
    
    return states_tensor, rewards_tensor, actions_tensor

def perform_training_step(model: nn.Module, 
                         states: torch.Tensor, 
                         actions_probs: torch.Tensor,
                         rewards: torch.Tensor,
                         learning_rate: float = 0.001,
                         beta: float = 0.2,
                         gamma: float = 0.9 ) -> Dict[str, float]:

    actions = torch.argmax(actions_probs, dim=1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    probs = model(states)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_probs = m.log_prob(actions)
    entropy = m.entropy()
    all_returns = []

    R = 0
    
   # Calculate discounted returns (vectorized)
    ep_returns = torch.zeros_like(rewards)
    ep_returns[-1] = rewards[-1]
    for t in reversed(range(len(rewards) - 1)):
        ep_returns[t] = rewards[t] + gamma * ep_returns[t + 1]

    # Normalize returns
    ep_returns = torch.tensor(ep_returns, dtype=torch.float32)
    ep_returns = (ep_returns - ep_returns.mean()) / (ep_returns.std() + 1e-5)
    
    # Combine policy loss with entropy bonus
    loss = -(log_probs * ep_returns).sum() - beta * entropy.sum()
    if torch.isnan(loss):
        raise ValueError("Loss is NaN!")
    

    # Perform backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate metrics
    with torch.no_grad():
        # Average reward
        avg_reward = rewards.mean()
        
        # Policy confidence (max action probability)
        max_action_prob = actions_probs.max(dim=-1)[0].mean()
        
        metrics = {
            'loss': loss.item(),
            'entropy_mean': entropy.mean().item(),
            'avg_reward': avg_reward.item(),
            'max_action_prob': max_action_prob.item(),
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
                     ) -> str:
    """Save training metrics to a JSON log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.json"
    log_path = os.path.join(folder_path, log_filename)
    
    log_data = {
        'timestamp': timestamp,
        #'original_model_info': original_info,
        'training_metrics': metrics
    }
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    logging.info(f"Saved training log to: {log_path}")
    return log_path

def find_latest_files(folder_path: str, snake_id: str) -> Dict[str, Optional[str]]:
    """Find the latest model, history and weights files in the folder."""
    files = {
        'model': None,
        'history': None,
        'state': None
    }
    
    # Find latest model file
    model_pattern = os.path.join(folder_path, f"snake_model_{snake_id}*.onnx")
    model_files = glob.glob(model_pattern)
    if model_files:
        model_files.sort(key=lambda x: os.path.getctime(x))
        files['model'] = model_files[-1]
        logging.info(f"Found latest model: {files['model']}")
    
    # Find latest history file
    history_pattern = os.path.join(folder_path, f"history_{snake_id}*.pkl")
    history_files = glob.glob(history_pattern)
    if history_files:
        history_files.sort(key=lambda x: os.path.getctime(x))
        files['history'] = history_files[-1]
        logging.info(f"Found latest history: {files['history']}")
    
    state_pattern = os.path.join(folder_path, f"snake_model_{snake_id}*.pth")
    state_files = glob.glob(history_pattern)
    if state_files:
        state_files.sort(key=lambda x: os.path.getctime(x))
        files['state'] = history_files[-1]
        logging.info(f"Found latest state: {files['state']}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Snake Agent Training Script")
    parser.add_argument("--snake_id", required=True, help="Agent ID")
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
        files = find_latest_files(args.folder, args.snake_id)
        
        if not files['model']:
            raise ValueError("No model file found in folder")
        if not files['history']:
            raise ValueError("No history file found in folder")
        if not files['state']:
            raise ValueError("No state file found in folder")
        #if not files['weights']:
        #    raise ValueError("No weights file found in folder")
        
        # Load model architecture
        model, optimizer, epoch = load_model(files['model'], files['state'])
        
        # Load and apply weights
        #weights = load_weights_from_file(files['weights'])
        #apply_weights_to_model(model, weights)
        
        # Load history
        history = load_history_from_file(files['history'])
        
        # Process history for training
        states, rewards, actions = process_history_for_training(history)
        # Perform training step
        logging.info("Performing training step...")
        metrics = perform_training_step(model, states, actions, rewards, args.learning_rate)
        
        # Log results
        logging.info("Training step completed!")
        logging.info(f"Metrics: {metrics}")
        
        # Save updated model
        updated_model_path, state_path = save_model(
                                            model = model, 
                                            input_size = states.shape[1],
                                            optimizer = optimizer, 
                                            epoch = epoch,
                                            snake_id = args.snake_id,
                                            model_save_dir = args.folder
                                            )
        
        # Save training log
        log_path = save_training_log(args.folder, metrics)
        
        logging.info("Training completed successfully!")
        logging.info(f"Updated model saved to: {updated_model_path}")
        logging.info(f"Training log saved to: {log_path}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.error(f"Traceback:{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
