import requests
import time
import json
import argparse
import logging
import sys
from snake_agent import NeuralSnakeAgent


def setup_logger(log_file: str):
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_state_stream(base_url):
    """Чтение состояния змейки как json по строкам."""
    response = requests.get(base_url, stream=True)
    for line in response.iter_lines():
        if line:
            decoded = line.decode()
            try:
                data = json.loads(decoded)
                yield data
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse JSON: {decoded}")


def send_move(move_url, move: str):
    """Отправка управляющего действия."""
    payload = {"move": move}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(move_url, json=payload, headers=headers)
        response.raise_for_status()
        logging.info(f"Sent move: {move}")
    except requests.RequestException as e:
        logging.error(f"Error sending move: {e}")


def neural_agent(snake_id: str, log_file: str, env_host: str, 
                model_save_dir: str = "models", learning_rate: float = 0.001):
    """Основная функция нейронного агента."""
    setup_logger(log_file)
    base_url = f"http://{env_host}/snake/{snake_id}"
    move_url = f"{base_url}/move"
    
    # Создаем агента
    agent = NeuralSnakeAgent(snake_id, model_save_dir, learning_rate)
    
    # Выводим информацию о модели
    model_info = agent.get_model_info()
    logging.info(f"Agent initialized: {model_info}")
    
    logging.info(f"Starting neural agent for snake_id={snake_id}")
    previous_action = "forward" 
    try:
        for data in get_state_stream(base_url):
            logging.info(f"Current state: {data}")
            
            # Сохраняем опыт
            reward = data.get("reward", 0)
            agent.save_experience(data, reward, previous_action)
            
            if data.get("game_over"):
                logging.info("Game over. Saving model and history...")
                saved_files = agent.save_all_data()
                logging.info(f"Saved files: {saved_files}")
                break
            
            # Предсказываем действие
            action = agent.predict_action(data)
              
            if action != "forward":
                send_move(move_url, action)
            previous_action = action
            
            time.sleep(0.2)
    
    except KeyboardInterrupt:
        logging.info("Agent interrupted by user.")
        # Сохраняем данные при прерывании
        try:
            saved_files = agent.save_all_data()
            logging.info(f"Data saved after interruption: {saved_files}")
        except Exception as e:
            logging.error(f"Error saving data after interruption: {e}")
    
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        # Все равно сохраняем данные при ошибке
        try:
            saved_files = agent.save_all_data()
            logging.info(f"Data saved after error: {saved_files}")
        except Exception as save_error:
            logging.error(f"Error saving data after error: {save_error}")
        raise


def load_saved_model(model_path: str):
    """Функция для загрузки сохраненной модели в другом коде."""
    import torch
    from snake_model import ModelManager
    
    manager = ModelManager()
    model, optimizer, info = manager.load_model(model_path)
    return model, optimizer, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Snake Agent")
    parser.add_argument("--snake_id", required=True, help="ID of the snake")
    parser.add_argument("--env_host", type=str, required=True, help="Host of the snake game")
    parser.add_argument("--log_file", default="neural_agent.log", help="Path to the log file")
    parser.add_argument("--model_save_dir", default="models", help="Directory to save models")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    
    args = parser.parse_args()
    neural_agent(args.snake_id, args.log_file, args.env_host, 
                args.model_save_dir, args.learning_rate)
