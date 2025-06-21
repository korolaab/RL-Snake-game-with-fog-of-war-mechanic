import torch
import logging
from snake_model import ModelManager
from state_processor import StateProcessor
from data_manager import DataManager


class NeuralSnakeAgent:
    """Основной класс агента змейки с нейронной сетью."""
    
    def __init__(self, snake_id: str, model_save_dir: str = "models", learning_rate: float = 0.001):
        self.snake_id = snake_id
        self.model_save_dir = model_save_dir
        self.learning_rate = learning_rate
        
        # Инициализация компонентов
        self.model_manager = ModelManager(model_save_dir)
        self.state_processor = StateProcessor()
        self.data_manager = DataManager(model_save_dir)
        
        # Модель и оптимизатор
        self.model = None
        self.optimizer = None
        self.model_initialized = False
        self.model_info = None
        
        # Возможные действия
        #TODO saving actions encoding to model file
        self.actions = ["left", "right", "forward"]
        
        # Пытаемся загрузить существующую модель
        self.load_existing_model()
    
    def load_existing_model(self):
        """Загрузка существующей модели при инициализации."""
        try:
            model, optimizer, model_info = self.model_manager.load_latest_model(
                self.snake_id, self.learning_rate
            )
            
            if model is not None:
                self.model = model
                self.optimizer = optimizer
                self.model_info = model_info
                self.model_initialized = True
                
                # Предупреждение если модель от другой змейки
                if model_info['snake_id'] != self.snake_id and model_info['snake_id'] != 'unknown':
                    logging.warning(f"Loaded model was trained for different snake_id: {model_info['snake_id']}")
            else:
                logging.info("No existing models found. Will create new model on first state.")
                
        except Exception as e:
            logging.error(f"Error loading existing model: {e}")
            logging.info("Will create new model on first state.")
    
    def ensure_model_initialized(self, input_size: int):
        """Убеждаемся что модель инициализирована с правильными размерами."""
        if not self.model_initialized:
            # Создаем новую модель
            self.model, self.optimizer = self.model_manager.create_new_model(
                input_size, self.learning_rate
            )
            self.model_initialized = True
        else:
            # Проверяем совместимость размеров
            if not self.model_manager.validate_input_size(self.model, input_size):
                actual_size = self.model_manager.get_model_input_size(self.model)
                logging.warning(f"Input size mismatch. Expected: {actual_size}, Got: {input_size}")
                logging.info("Recreating model with correct input size...")
                
                self.model, self.optimizer = self.model_manager.create_new_model(
                    input_size, self.learning_rate
                )
    
    def predict_action(self, state):
        """Предсказание действия на основе состояния."""
        try:
            # Обработка состояния
            state_tensor = self.state_processor.process_state(state)
            
            # Преобразуем в плоский вектор для подачи в модель
            flat_tensor = state_tensor.flatten().unsqueeze(0)  # добавляем batch dimension
            input_size = flat_tensor.shape[1]
            
            # Убеждаемся что модель инициализирована
            self.ensure_model_initialized(input_size)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(flat_tensor)
                action_probs = torch.softmax(outputs, dim=1)
                action_idx = torch.argmax(action_probs, dim=1).item()
            
            predicted_action = self.actions[action_idx]
            logging.debug(f"Predicted action: {predicted_action} (probs: {action_probs.numpy()})")
            
            return predicted_action
            
        except Exception as e:
            logging.error(f"Error predicting action: {e}")
            # Возвращаем случайное действие в случае ошибки
            import random
            return random.choice(self.actions)
    
    def save_experience(self, state, reward, action):
        self.data_manager.add_experience(state, reward, action)
    
    def save_all_data(self):

        try:
            saved_files = {}
            
            # Сохраняем модель
            if self.model is not None:
                model_path = self.model_manager.save_model(
                    self.model, self.optimizer, self.snake_id
                )
                saved_files['model'] = model_path
            
            # Сохраняем историю и веса
            data_files = self.data_manager.save_all_data(self.model, self.snake_id)
            saved_files.update(data_files)
            
            # Выводим статистику
            stats = self.data_manager.get_statistics()
            logging.info(f"Game completed. Statistics: {stats}")
            
            return saved_files
            
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            raise
    
    def get_model_info(self):
        """Получение информации о текущей модели."""
        info = {
            'model_initialized': self.model_initialized,
            'snake_id': self.snake_id,
            'actions': self.actions,
            'learning_rate': self.learning_rate
        }
        
        if self.model is not None:
            info['input_size'] = self.model_manager.get_model_input_size(self.model)
        
        if self.model_info is not None:
            info['loaded_from'] = self.model_info
        
        return info
    
    def get_statistics(self):
        """Получение статистики по накопленным данным."""
        return self.data_manager.get_statistics()
