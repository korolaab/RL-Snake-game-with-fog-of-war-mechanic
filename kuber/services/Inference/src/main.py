import requests
import time
import json
import argparse
import sys
import asyncio
import os
from snake_agent import GRPCSnakeAgent
from datetime import datetime
import threading
import logging
import logger


class StreamReader:
    def __init__(self, base_url):
        self.base_url = base_url
        self.latest_state = None
        self.latest_timestamp = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.new_state_event = threading.Event()
        self.last_seen_frame = None  # (episode, frame) tuple to prevent duplicates

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
        """Block until new state is available, ignore duplicate (episode, frame) states"""
        while True:
            if self.new_state_event.wait(timeout):
                with self.lock:
                    if self.latest_state is not None:
                        state = self.latest_state
                        timestamp = self.latest_timestamp
                        episode = state.get('episode')
                        frame = state.get('frame')
                        key = (episode, frame)
                        # Only yield if this (episode, frame) is not duplicate
                        if key != self.last_seen_frame:
                            logging.info({"event": "agent_new_state", "episode": episode, "frame": frame, "state": state})
                            self.last_seen_frame = key
                            self.latest_state = None
                            self.new_state_event.clear()
                            return state, timestamp
                        else:
                            logging.info({"event": "agent_duplicate_state_skipped", "episode": episode, "frame": frame})
                            # Skip duplicate state, clear flag, and wait for next
                            self.latest_state = None
                            self.new_state_event.clear()
            else:
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
                            logging.info({"event": "game_over_detected", "source": "stream"})
                            break
                            
                    except json.JSONDecodeError:
                        logging.warning({"event": "failed_to_parse_json", "data": decoded})

        except Exception as e:
            logging.error({"event": "stream_reading_error", "exception": e})
        finally:
            self.running = False

def send_move(move_url, move: str):
    """Send control action."""
    payload = {"move": move}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(move_url, json=payload, headers=headers)
        response.raise_for_status()
        logging.debug({"event": "sent_move", "move": move})
    except requests.RequestException as e:
        logging.error({"event": "error_sending_move", "exception": e})


async def neural_agent_grpc(snake_id: str, log_file: str, env_host: str,
                           model_save_dir: str = "models", learning_rate: float = 0.001,
                           grpc_host: str = "localhost", grpc_port: int = 50051,
                           batch_size: int = 5):
    """
    Neural agent with gRPC communication to training service.
    batch_size = number of episodes before sending to training
    """
    base_url = f"http://{env_host}/snake/{snake_id}"
    move_url = f"{base_url}/move"
    reset_url = f"http://{env_host}/reset"

    logging.info({"event": "starting_grpc_neural_agent", "snake_id": snake_id})
    logging.info({"event": "batch_size_configured", "batch_size": batch_size, "unit": "episodes"})
    logging.info({"event": "grpc_training_service_configured", "host": grpc_host, "port": grpc_port})
    logging.info({"event": "mode_configured", "mode": "collect_episodes_then_batch_train"})

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
    logging.info({"event": "agent_initialized", "model_info": model_info})

    try:
        episode_count = 0

        while True:  # Outer loop for multiple games
            episode_count += 1
            logging.info({"event": "starting_episode", "episode_count": episode_count})

            previous_action = "forward"
            stream_reader = StreamReader(base_url)
            stream_reader.start()

            try:
                # Inner loop for one episode (game)
                while True:
                    data, tech_timestamp = stream_reader.get_latest_state()

                    if data is None or tech_timestamp is None:
                        logging.warning({"event": "no_data_received_from_stream"})
                        continue

                    send_datetime_str = data.get("datetime")
                    send_timestamp = datetime.fromisoformat(send_datetime_str).timestamp()
                    delay = tech_timestamp - send_timestamp

                    logging.info({"event": "state_received", 
                                  "episode": data.get("episode",'null'),
                                  "frame": data.get("frame",'null'),
                                  "reward": data.get("reward", 0), "delay": f"{delay:.3f}s"})

                    # Add experience to current episode
                    should_send_batch = agent.add_experience(
                        state=data,
                        action=previous_action,
                        reward=data.get("reward", 0),
                        done=data.get("game_over", False)
                    )

                    # If episode ended, check if we should send batch
                    if data.get("game_over"):
                        logging.info({"event": "episode_ended", "episode_count": episode_count})

                        # Check if we have enough episodes for a batch
                        if should_send_batch:
                            logging.info({"event": "sending_batch", "episodes_completed": agent.batch_size})
                            success = await agent.send_training_batch_and_wait()
                            if success:
                                logging.info({"event": "received_improved_model", "source": "batch_training"})
                            else:
                                logging.warning({"event": "failed_to_get_model_update", "action": "continuing_with_current_model"})
                        else:
                            completed_episodes = len(agent.completed_episodes)
                            remaining = agent.batch_size - completed_episodes
                            logging.info(f"📊 Episode added to batch ({completed_episodes}/{agent.batch_size}). "
                                       f"Need {remaining} more episodes before training.")
                        
                        # Reset environment for next episode
                        try:
                            reset_response = requests.post(reset_url, timeout=5)
                            if reset_response.status_code == 200:
                                logging.info({"event": "environment_reset_successful"})
                            else:
                                logging.warning({"event": "reset_failed", "status_code": reset_response.status_code})
                        except Exception as reset_error:
                            logging.error({"event": "error_resetting_environment", "exception": reset_error})
                        
                        break  # Exit inner loop (end of episode)
                    
                    # Predict action for next step
                    action = agent.predict_action(data)
                      
                    if action != "forward":
                        send_move(move_url, action)
                    previous_action = action
                    
            except Exception as game_error:
                logging.error({"event": "error_during_episode", "episode_count": episode_count, "exception": game_error})
                # Save data even on error
                try:
                    # Send any remaining episodes if we have some
                    if len(agent.completed_episodes) > 0:
                        logging.info({"event": "sending_partial_batch_due_to_error", "episode_count": len(agent.completed_episodes)})
                        await agent.send_training_batch_and_wait()
                    saved_files = agent.save_all_data()
                    logging.info({"event": "data_saved_after_episode_error", "saved_files": saved_files})
                except Exception as save_error:
                    logging.error({"event": "error_saving_data_after_episode_error", "exception": save_error})
            finally:
                stream_reader.stop()
                
    except KeyboardInterrupt:
        logging.info({"event": "keyboard_interrupt", "action": "shutting_down"})
    finally:
        await agent.shutdown()
        logging.info({"event": "agent_shutdown_complete"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural network agent with gRPC")
    parser.add_argument("--snake-id", type=str, required=True, help="Snake ID for this agent")
    parser.add_argument("--log-file", type=str, default="agent_log.json", help="Log file path")
    parser.add_argument("--env-host", type=str, default="localhost:5000", help="Environment host URL")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--grpc-host", type=str, default="localhost", help="gRPC server host")
    parser.add_argument("--grpc-port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--batch-size", type=int, default=5, help="Episodes per batch")

    args = parser.parse_args()

    # Configure logging
    logger.setup_as_default()

    # Run the agent
    asyncio.run(neural_agent_grpc(
        snake_id=args.snake_id,
        log_file=args.log_file,
        env_host=args.env_host,
        model_save_dir=args.model_dir,
        learning_rate=args.learning_rate,
        grpc_host=args.grpc_host,
        grpc_port=args.grpc_port,
        batch_size=args.batch_size
    ))
