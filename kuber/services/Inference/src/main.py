import requests
import time
import json
import argparse
import logging
import sys
import asyncio
import os
from snake_agent import GRPCSnakeAgent
from datetime import datetime
import threading


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
        if self.new_state_event.wait(timeout):
            with self.lock:
                if self.latest_state is not None:
                    state = self.latest_state
                    timestamp = self.latest_timestamp
                    self.latest_state = None
                    self.new_state_event.clear()
                    return state, timestamp
        
        return None, None

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
                        
                        with self.lock:
                            self.latest_state = data
                            self.latest_timestamp = time.time()
                        
                        self.new_state_event.set()

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
    """Send control action."""
    payload = {"move": move}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(move_url, json=payload, headers=headers)
        response.raise_for_status()
        logging.info(f"Sent move: {move}, {response.text}")
    except requests.RequestException as e:
        logging.error(f"Error sending move: {e}")


async def neural_agent_grpc(snake_id: str, log_file: str, env_host: str, 
                           model_save_dir: str = "models", learning_rate: float = 0.001,
                           grpc_host: str = "localhost", grpc_port: int = 50051,
                           batch_size: int = 50):
    """
    Neural agent with gRPC communication to training service.
    """
    setup_logger(log_file)
    base_url = f"http://{env_host}/snake/{snake_id}"
    move_url = f"{base_url}/move"
    reset_url = f"http://{env_host}/reset"
    
    logging.info(f"🐍 Starting gRPC neural agent for snake_id={snake_id}")
    logging.info(f"📦 Batch size: {batch_size}")
    logging.info(f"📡 gRPC Training Service: {grpc_host}:{grpc_port}")
    logging.info("⏳ Mode: Wait for model updates before continuing")
    
    # Create agent
    agent = GRPCSnakeAgent(
        snake_id=snake_id, 
        model_save_dir=model_save_dir, 
        learning_rate=learning_rate,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        batch_size=batch_size
    )
    
    # Connect to training service
    await agent.connect_to_training_service()
    
    # Output model info
    model_info = agent.get_model_info()
    logging.info(f"🤖 Agent initialized: {model_info}")
    
    try:
        while True:  # Outer loop for multiple games
            logging.info(f"🎮 Starting new game (Episode {agent.current_episode + 1})...")
            
            previous_action = "forward"
            stream_reader = StreamReader(base_url)
            stream_reader.start()
            
            try:
                # Inner loop for one game
                while True:
                    data, tech_timestamp = stream_reader.get_latest_state()
                    
                    if data is None or tech_timestamp is None:
                        logging.warning("No data received from stream")
                        continue
                    
                    send_datetime_str = data.get("datetime") 
                    send_timestamp = datetime.fromisoformat(send_datetime_str).timestamp() 
                    delay = tech_timestamp - send_timestamp
                    
                    logging.info(f"📍 State: reward={data.get('reward', 0)}, delay={delay:.3f}s")
                    
                    # Add experience to buffer
                    should_send_batch = agent.add_experience(
                        state=data,
                        action=previous_action,
                        reward=data.get("reward", 0),
                        done=data.get("game_over", False)
                    )
                    
                    # Send batch and wait for improved model
                    if should_send_batch:
                        logging.info("⏳ Sending batch via gRPC and waiting for improved model...")
                        success = await agent.send_training_batch_and_wait()
                        if success:
                            logging.info("✅ Received improved model, continuing with better weights!")
                        else:
                            logging.warning("⚠️ Failed to get model update, continuing with current model")
                    
                    if data.get("game_over"):
                        logging.info("💀 Game over!")
                        
                        # Send remaining experiences if any
                        if len(agent.experience_buffer) > 0:
                            logging.info(f"📦 Sending final batch: {len(agent.experience_buffer)} experiences")
                            success = await agent.send_training_batch_and_wait()
                            if success:
                                logging.info("✅ Received final improved model!")
                            else:
                                logging.warning("⚠️ Failed to get final model update")
                        
                        # Save data locally (backup)
                        saved_files = agent.save_all_data()
                        logging.info(f"💾 Saved backup files: {saved_files}")
                        
                        # Reset environment
                        try:
                            reset_response = requests.post(reset_url, timeout=5)
                            if reset_response.status_code == 200:
                                logging.info("🔄 Environment reset successful")
                            else:
                                logging.warning(f"Reset failed with status: {reset_response.status_code}")
                        except Exception as reset_error:
                            logging.error(f"Error resetting environment: {reset_error}")
                        
                        break  # Exit inner loop
                    
                    # Predict action
                    action = agent.predict_action(data)
                      
                    if action != "forward":
                        send_move(move_url, action)
                    previous_action = action
                    
            except Exception as game_error:
                logging.error(f"Error during game: {game_error}")
                # Save data even on error
                try:
                    if len(agent.experience_buffer) > 0:
                        await agent.send_training_batch_and_wait()
                    saved_files = agent.save_all_data()
                    logging.info(f"Data saved after game error: {saved_files}")
                except Exception as save_error:
                    logging.error(f"Error saving data after game error: {save_error}")
            finally:
                stream_reader.stop()
                
    except KeyboardInterrupt:
        logging.info("Agent interrupted by user.")
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        raise
    finally:
        # Disconnect from training service
        await agent.disconnect_from_training_service()


def neural_agent(snake_id: str, log_file: str, env_host: str, 
                model_save_dir: str = "models", learning_rate: float = 0.001,
                grpc_host: str = "localhost", grpc_port: int = 50051,
                batch_size: int = 50):
    """
    Synchronous wrapper for running async gRPC agent.
    """
    try:
        asyncio.run(neural_agent_grpc(
            snake_id=snake_id,
            log_file=log_file,
            env_host=env_host,
            model_save_dir=model_save_dir,
            learning_rate=learning_rate,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            batch_size=batch_size
        ))
    except KeyboardInterrupt:
        logging.info("Agent interrupted by user.")
    except Exception as e:
        logging.error(f"Error in neural agent: {e}")
        raise


def load_saved_model(model_path: str):
    """Function to load saved model in other code."""
    import torch
    from snake_model import ModelManager
    
    manager = ModelManager()
    model, optimizer, info = manager.load_model(model_path)
    return model, optimizer, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Snake Agent with gRPC Training")
    parser.add_argument("--snake_id", required=True, help="ID of the snake")
    parser.add_argument("--env_host", type=str, required=True, help="Host of the snake game")
    parser.add_argument("--log_file", default="neural_agent.log", help="Path to the log file")
    parser.add_argument("--model_save_dir", default="models", help="Directory to save models")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--grpc_host", default="localhost", help="gRPC training service host")
    parser.add_argument("--grpc_port", type=int, default=50051, help="gRPC training service port")
    parser.add_argument("--batch_size", type=int, default=50, help="Number of experiences before sending to training")
    
    args = parser.parse_args()
    neural_agent(
        snake_id=args.snake_id, 
        log_file=args.log_file, 
        env_host=args.env_host,
        model_save_dir=args.model_save_dir, 
        learning_rate=args.learning_rate,
        grpc_host=args.grpc_host,
        grpc_port=args.grpc_port,
        batch_size=args.batch_size
    )
