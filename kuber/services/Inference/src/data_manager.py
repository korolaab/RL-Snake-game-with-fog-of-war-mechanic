import pickle
import os
import logging
import numpy as np
from datetime import datetime
from collections import deque


class DataManager:
    """Менеджер для сохранения и загрузки данных обучения."""
    
    def __init__(self, save_dir: str = "models"):
        self.save_dir = save_dir
        self.history = []  # история (state, reward)
        os.makedirs(save_dir, exist_ok=True)
    
    def add_experience(self, state, reward, action):
        """Добавление нового опыта в историю."""
        experience = {
            'state': state,
            'reward': reward,
            'action': action,
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(experience)
        logging.debug({"message": f"Added experience: reward={reward}, total_experiences={len(self.history)}"})
    
    def get_history(self):
        """Получение всей истории."""
        return self.history.copy()
    
    def clear_history(self):
        """Очистка истории."""
        self.history.clear()
        logging.info({"message": "History cleared"})
    
    def save_history(self, snake_id: str, filename_suffix: str = None):
        """Сохранение истории в файл."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename_suffix:
            filename = f"history_{snake_id}_{filename_suffix}_{timestamp}.pkl"
        else:
            filename = f"history_{snake_id}_{timestamp}.pkl"
        
        history_path = os.path.join(self.save_dir, filename)
        
        try:
            with open(history_path, 'wb') as f:
                pickle.dump(self.history, f)
            logging.info({"message": f"History saved to: {history_path} ({len(self.history)} experiences)"})
            return history_path
        except Exception as e:
            logging.error({"message": f"Error saving history: {e}"})
            raise
    
    def load_history(self, history_path: str):
        """Загрузка истории из файла."""
        try:
            with open(history_path, 'rb') as f:
                self.history = pickle.load(f)
            logging.info({"message": f"History loaded from: {history_path} ({len(self.history)} experiences)"})
            return self.history
        except Exception as e:
            logging.error({"message": f"Error loading history from {history_path}: {e}"})
            raise
    
    def save_model_weights(self, model, snake_id: str, filename_suffix: str = None):
        """Сохранение весов модели отдельно в numpy формате."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename_suffix:
            filename = f"weights_{snake_id}_{filename_suffix}_{timestamp}.pkl"
        else:
            filename = f"weights_{snake_id}_{timestamp}.pkl"
        
        weights_path = os.path.join(self.save_dir, filename)
        
        try:
            weights = {}
            for name, param in model.named_parameters():
                weights[name] = param.data.numpy()
            
            with open(weights_path, 'wb') as f:
                pickle.dump(weights, f)
            
            logging.info({"message": f"Model weights saved to: {weights_path}"})
            return weights_path
        except Exception as e:
            logging.error({"message": f"Error saving weights: {e}"})
            raise
    
    def load_model_weights(self, weights_path: str):
        """Загрузка весов модели из файла."""
        try:
            with open(weights_path, 'rb') as f:
                weights = pickle.load(f)
            logging.info({"message": f"Model weights loaded from: {weights_path}"})
            return weights
        except Exception as e:
            logging.error({"message": f"Error loading weights from {weights_path}: {e}"})
            raise
    
    def get_statistics(self):
        """Получение статистики по накопленным данным."""
        if not self.history:
            return {
                'total_experiences': 0,
                'total_reward': 0,
                'average_reward': 0,
                'min_reward': 0,
                'max_reward': 0
            }
        
        rewards = [exp['reward'] for exp in self.history]
        
        return {
            'total_experiences': len(self.history),
            'total_reward': sum(rewards),
            'average_reward': sum(rewards) / len(rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'unique_rewards': len(set(rewards))
        }
    
    def save_all_data(self, model, snake_id: str, filename_suffix: str = None):
        """Сохранение всех данных: истории и весов."""
        saved_files = {}
        
        try:
            # Сохраняем историю
            history_path = self.save_history(snake_id, filename_suffix)
            saved_files['history'] = history_path
            
            # Сохраняем веса
            weights_path = self.save_model_weights(model, snake_id, filename_suffix)
            saved_files['weights'] = weights_path
            
            # Выводим статистику
            stats = self.get_statistics()
            logging.info({"message": f"Data statistics: {stats}"})
            
            return saved_files
            
        except Exception as e:
            logging.error({"message": f"Error saving all data: {e}"})
            raise
