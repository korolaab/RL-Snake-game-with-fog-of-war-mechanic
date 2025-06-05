import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import logging
from datetime import datetime


class SnakeNet(nn.Module):
    """Нейронная сеть для змейки."""
    
    def __init__(self, input_size):
        super(SnakeNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(16, 3)
        )
    
    def forward(self, x):
        return self.network(x)


class ModelManager:
    """Менеджер для создания, загрузки и сохранения моделей."""
    
    def __init__(self, model_save_dir: str = "models"):
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
    
    def create_new_model(self, input_size: int, learning_rate: float = 0.001):
        """Создание новой модели с нуля."""
        model = SnakeNet(input_size)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        logging.info(f"Created new model with input size: {input_size}")
        return model, optimizer
    
    def find_latest_model(self, snake_id: str = None):
        """Поиск самой новой модели для данной змейки или любой змейки."""
        # Сначала ищем модели для конкретной змейки
        if snake_id:
            pattern = os.path.join(self.model_save_dir, f"snake_model_{snake_id}_*.pth")
        else:
            pattern = os.path.join(self.model_save_dir, "snake_model_*.pth")
        
        model_files = glob.glob(pattern)
        
        if not model_files:
            # Если нет моделей для конкретной змейки, ищем любые модели
            if snake_id:
                logging.info(f"No models found for snake_id={snake_id}, searching for any models...")
                return self.find_latest_model(snake_id=None)
            else:
                logging.info("No saved models found.")
                return None
        
        # Сортируем по времени создания файла (самый новый последний)
        model_files.sort(key=lambda x: os.path.getctime(x))
        latest_model = model_files[-1]
        
        logging.info(f"Found latest model: {latest_model}")
        return latest_model
    
    def load_model(self, model_path: str, learning_rate: float = 0.001):
        """Загрузка модели из файла."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Восстанавливаем модель
            if 'model_architecture' in checkpoint:
                model = checkpoint['model_architecture']
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise ValueError("Model architecture not found in checkpoint")
            
            # Восстанавливаем оптимизатор
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            original_snake_id = checkpoint.get('snake_id', 'unknown')
            timestamp = checkpoint.get('timestamp', 'unknown')
            
            logging.info(f"Successfully loaded model from {model_path}")
            logging.info(f"Original snake_id: {original_snake_id}, timestamp: {timestamp}")
            
            return model, optimizer, {
                'snake_id': original_snake_id,
                'timestamp': timestamp,
                'model_path': model_path
            }
        
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def load_latest_model(self, snake_id: str, learning_rate: float = 0.001):
        """Загрузка самой новой модели для змейки."""
        latest_model_path = self.find_latest_model(snake_id)
        
        if latest_model_path is None:
            logging.info("No existing models found.")
            return None, None, None
        
        return self.load_model(latest_model_path, learning_rate)
    
    def save_model(self, model, optimizer, snake_id: str, additional_data: dict = None):
        """Сохранение модели в файл."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_save_dir, f"snake_model_{snake_id}_{timestamp}.pth")
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_architecture': model,
            'optimizer_state_dict': optimizer.state_dict(),
            'snake_id': snake_id,
            'timestamp': timestamp
        }
        
        # Добавляем дополнительные данные если есть
        if additional_data:
            checkpoint.update(additional_data)
        
        torch.save(checkpoint, model_path)
        logging.info(f"Model saved to: {model_path}")
        return model_path
    
    def get_model_input_size(self, model):
        """Получение размера входного слоя модели."""
        if hasattr(model, 'network') and hasattr(model.network[0], 'in_features'):
            return model.network[0].in_features
        return None
    
    def validate_input_size(self, model, expected_size: int):
        """Проверка совместимости размера входных данных с моделью."""
        actual_size = self.get_model_input_size(model)
        if actual_size is None:
            return False
        return actual_size == expected_size
