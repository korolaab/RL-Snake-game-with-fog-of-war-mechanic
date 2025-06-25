import requests
import time
import json
import argparse
import logging
import sys
from snake_agent import NeuralSnakeAgent
from datetime import datetime
import threading


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

class StreamReader:
    def __init__(self, base_url):
        self.base_url = base_url
        self.latest_state = None
        self.latest_timestamp = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.new_state_event = threading.Event()
        
    def start(self):
        """Start reading stream in background"""
        self.running = True
        self.thread = threading.Thread(target=self._read_stream, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop reading stream"""
        self.running = False
        if self.thread:
            self.thread.join()
            
            
    def get_latest_state(self, timeout=None):
        """Block until new state is available"""
        # Wait for new state signal
        if self.new_state_event.wait(timeout):
            with self.lock:
                if self.latest_state is not None:
                    state = self.latest_state
                    timestamp = self.latest_timestamp
                    self.latest_state = None  # Clear after returning
                    self.new_state_event.clear()  # Clear the event
                    return state, timestamp
        
        return None, None  # Timeout or no state

    def _read_stream(self):
        """Background thread that continuously reads stream"""
        try:
            response = requests.get(self.base_url, stream=True)
            
            for line in response.iter_lines():
                if not self.running:
                    break
                    
                if line:
                    try:
                        decoded = line.decode()
                        data = json.loads(decoded)
                        
                        # Update latest state atomically
                        with self.lock:
                            self.latest_state = data
                            self.latest_timestamp = time.time()
                        
                        self.new_state_event.set()

                        # Stop if game over
                        if data.get('game_over'):
                            logging.info("Game over detected in stream")
                            break
                            
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to parse JSON: {decoded}")
                        
        except Exception as e:
            logging.error(f"Stream reading error: {e}")
        finally:
            self.running = False

def send_move(move_url, move: str):
    """Отправка управляющего действия."""
    payload = {"move": move}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(move_url, json=payload, headers=headers)
        response.raise_for_status()
        logging.info(f"Sent move: {move}, {response.text}")
    except requests.RequestException as e:
        logging.error(f"Error sending move: {e}")


def neural_agent(snake_id: str, log_file: str, env_host: str, 
                model_save_dir: str = "models", learning_rate: float = 0.001):
    """
    Основная функция нейронного агента с асинхронным обучением.
    """
    setup_logger(log_file)
    base_url = f"http://{env_host}/snake/{snake_id}"
    move_url = f"{base_url}/move"
    reset_url = f"http://{env_host}/reset"  # URL для сброса среды
    
    
    
    logging.info(f"Starting neural agent for snake_id={snake_id}")
    
    try:
        while True:  # Внешний цикл для множественных игр
            # Загружаем последнюю модель перед началом новой игры
            # Создаем агента
            agent = NeuralSnakeAgent(snake_id, model_save_dir, learning_rate)
            # Выводим информацию о модели
            model_info = agent.get_model_info()
            logging.info(f"Agent initialized: {model_info}")
            logging.info("Loaded latest model for new episode")
            
            previous_action = "forward"
            stream_reader = StreamReader(base_url)
            stream_reader.start()
            data = None
            tech_timestamp = None
            
            try:
                # Внутренний цикл для одной игры
                while True:
                    data, tech_timestamp = stream_reader.get_latest_state()
                    
                    send_datetime_str = data.get("datetime") 
                    send_timestamp = datetime.fromisoformat(send_datetime_str).timestamp() 
                    delay = tech_timestamp - send_timestamp
                    logging.info(f"Current state: {data}")
                    logging.info(f"Delay: {delay}")
                    
                    # Сохраняем опыт
                    reward = data.get("reward", 0)
                    agent.save_experience(data, reward, previous_action)
                    
                    if data.get("game_over"):
                        logging.info("Game over. Saving experience data...")
                        saved_files = agent.save_all_data()
                        logging.info(f"Saved files: {saved_files}")
                        
                        # Сбрасываем среду немедленно (не ждем обучения)
                        try:
                            reset_response = requests.post(reset_url, timeout=5)
                            if reset_response.status_code == 200:
                                logging.info("Environment reset successful")
                            else:
                                logging.warning(f"Reset failed with status: {reset_response.status_code}")
                        except Exception as reset_error:
                            logging.error(f"Error resetting environment: {reset_error}")
                        
                        break  # Выходим из внутреннего цикла (одна игра закончена)
                    
                    # Предсказываем действие
                    action = agent.predict_action(data)
                      
                    if action != "forward":
                        send_move(move_url, action)
                    previous_action = action
                    
            except Exception as game_error:
                logging.error(f"Error during game: {game_error}")
                # Сохраняем данные даже при ошибке в игре
                try:
                    saved_files = agent.save_all_data()
                    logging.info(f"Data saved after game error: {saved_files}")
                except Exception as save_error:
                    logging.error(f"Error saving data after game error: {save_error}")
            finally:
                stream_reader.stop()
                
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
