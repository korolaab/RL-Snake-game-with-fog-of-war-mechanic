import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import logging
import onnx
from pathlib import Path
import onnx2torch
from datetime import datetime


class SnakeNet(nn.Module):
    """Нейронная сеть для змейки."""
    
    def __init__(self, input_size):
        super(SnakeNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(8, 3),
            nn.Softmax(dim=1)  # Fixed: added dim=1
        )
    
    def forward(self, x):
        return self.network(x)


class ModelManager:
    """Менеджер для создания, загрузки и сохранения моделей."""
    
    def __init__(self, model_save_dir: str = "models"):
        self.model_save_dir = model_save_dir
        self.input_size = None  # Initialize input_size
        os.makedirs(model_save_dir, exist_ok=True)
    
    def create_new_model(self, input_size: int, learning_rate: float = 0.001):
        """Создание новой модели с нуля."""
        self.input_size = input_size
        model = SnakeNet(self.input_size)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        logging.info({"event": "created_new_model", "input_size": input_size})
        return model, optimizer
    
    def find_latest_model(self, snake_id: str = None):
        """Search for the latest ONNX model for a specific snake or any snake."""      
        # First, search for models for a specific snake                             
        if snake_id:                                                            
            pattern = os.path.join(self.model_save_dir, f"snake_model_{snake_id}_*.onnx")
        else:                                                                   
            pattern = os.path.join(self.model_save_dir, "snake_model_*.onnx")    
                                                                                
        model_files = glob.glob(pattern)                                        
                                                                                
        if not model_files:                                                     
            # If no models found for specific snake, search for any models         
            if snake_id:                                                        
                logging.info({"event": "no_onnx_models_found_for_snake", "snake_id": snake_id, "action": "searching_for_any_models"})
                return self.find_latest_model(snake_id=None)                    
            else:                                                               
                logging.info({"event": "no_saved_onnx_models_found"})                          
                return None                                                     
                                                                                
        # Sort by file creation time (newest last)           
        model_files.sort(key=lambda x: os.path.getctime(x))                     
        latest_model = model_files[-1]                                          
                                                                                
        logging.info({"event": "found_latest_onnx_model", "model_path": latest_model})                     
        return latest_model

    def load_latest_model(self, snake_id: str, learning_rate: float = 0.001): 
        """Load the latest ONNX model for the snake and convert to PyTorch."""                           
        latest_model_path = self.find_latest_model(snake_id)                    
                                                                                
        if latest_model_path is None:                                           
            logging.info({"event": "no_existing_onnx_models_found"})                           
            return None, None, None                                             
                                                                                
        return self.load_onnx_model(latest_model_path, learning_rate)

    def load_onnx_model(self, model_path: str, learning_rate: float = 0.001):
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
            
            # Extract input size from model if possible
            try:
                # Try to get input size from model structure
                first_layer = None
                for module in pytorch_model.modules():
                    if isinstance(module, nn.Linear):
                        first_layer = module
                        break
                
                if first_layer:
                    self.input_size = first_layer.in_features
                    logging.info({"event": "detected_model_input_size", "input_size": self.input_size})
            except:
                logging.warning({"event": "could_not_detect_input_size"})
            
            # Check for accompanying state file
            state_path = Path(model_path).with_suffix('.pth')
            
            # Create model info dictionary
            model_info = {
                'snake_id': 'unknown',
                'timestamp': datetime.now().isoformat(),
                'model_path': model_path,
                'input_size': self.input_size
            }
            
            if state_path.exists():
                try:
                    checkpoint = torch.load(state_path, map_location='cpu')
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'snake_id' in checkpoint:
                        model_info['snake_id'] = checkpoint['snake_id']
                    if 'timestamp' in checkpoint:
                        model_info['timestamp'] = checkpoint['timestamp']
                    logging.info({"event": "loaded_training_state", "state_path": state_path})
                except Exception as e:
                    logging.warning({"event": "could_not_load_training_state", "exception": e})
            
            logging.info({"event": "successfully_loaded_onnx_model", "model_path": model_path})
            
            return pytorch_model, optimizer, model_info
            
        except Exception as e:
            logging.error({"event": "failed_to_load_onnx_model", "model_path": model_path, "exception": e})
            logging.error({"event": "error_details", "details": str(e)})
            return None, None, None 

    def save_model(self, model, optimizer, snake_id: str, additional_data: dict = None):
        """Save PyTorch model as ONNX format with additional training state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ONNX model path
        onnx_model_path = os.path.join(self.model_save_dir, f"snake_model_{snake_id}_{timestamp}.onnx")
        
        # Training state path (for optimizer and additional data)
        state_path = os.path.join(self.model_save_dir, f"snake_model_state_{snake_id}_{timestamp}.pth")
        
        try:
            # Set model to evaluation mode for export
            model.eval()
            
            # Determine input size
            input_size = self.input_size
            if input_size is None:
                # Try to get from model
                input_size = self.get_model_input_size(model)
                if input_size is None:
                    input_size = 64  # Default fallback
                    logging.warning({"event": "using_fallback_input_size", "input_size": input_size})
            
            # Create dummy input for tracing
            dummy_input = torch.randn(1, input_size)
            
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
                'model_input_shape': list(dummy_input.shape),
                'input_size': input_size
            }
            
            # Add additional data if provided
            if additional_data:
                checkpoint.update(additional_data)
            
            torch.save(checkpoint, state_path)
            
            # Verify ONNX model
            onnx_model = onnx.load(onnx_model_path)
            onnx.checker.check_model(onnx_model)
            
            logging.info({"event": "onnx_model_saved", "path": onnx_model_path})
            logging.info({"event": "training_state_saved", "path": state_path})
            
            # Set model back to training mode
            model.train()
            
            return onnx_model_path
        
        except Exception as e:
            logging.error({"event": "failed_to_save_model_as_onnx", "exception": e})
            return None
    
    def get_model_input_size(self, model):
        """Получение размера входного слоя модели."""
        try:
            # For SnakeNet models
            if hasattr(model, 'network') and hasattr(model.network[0], 'in_features'):
                return model.network[0].in_features
            
            # For converted ONNX models - find first Linear layer
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    return module.in_features
            
            return None
        except:
            return None
    
    def validate_input_size(self, model, expected_size: int):
        """Проверка совместимости размера входных данных с моделью."""
        actual_size = self.get_model_input_size(model)
        logging.debug({"event": "validate_input_size", 
                       "actual_size": actual_size,
                       "expected_size": expected_size})
        if actual_size is None:
            return False
        return actual_size == expected_size
