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
import socket


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


def connect_sync_socket(sync_host, sync_port):      
    try:
        # Detailed logging before connection attempt
        logging.debug({
            "event": "sync_socket_connection_start", 
            "host": sync_host, 
            "port": sync_port,
            "current_time": str(datetime.now())
        })

        # Create socket with more robust configuration
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(10)  # 10-second timeout for connection

        try:
            # Attempt connection
            sock.connect((sync_host, sync_port))
            
            # Log successful connection details
            logging.debug({
                "event": "sync_socket_connected", 
                "local_address": sock.getsockname(),
                "remote_address": sock.getpeername()
            })
            
            # Send identification with extra logging
            identification = b'INF\n'
            logging.debug({
                "event": "sending_client_identification", 
                "identification_bytes": identification,
                "identification_str": identification.decode().strip()
            })
            
            send_result = sock.send(identification)
            logging.debug({
                "event": "client_identification_sent", 
                "bytes_sent": send_result
            })
            
            # Reset timeout for subsequent operations
            sock.settimeout(None)
            
            # Create file-like object for easier reading
            sock_file = sock.makefile('rb')
            
            logging.info({
                "event": "sync_reconnect", 
                "status": "success (TCP)", 
                "host": sync_host, 
                "port": sync_port,
                "connection_details": {
                    "local_address": sock.getsockname(),
                    "remote_address": sock.getpeername()
                }
            })
            
            return sock, sock_file
        
        except (socket.timeout, socket.error) as conn_error:
            logging.error({
                "event": "sync_connection_error", 
                "error": str(conn_error),
                "error_type": type(conn_error).__name__,
                "host": sync_host, 
                "port": sync_port
            })
            raise
    
    except Exception as e:
        logging.error({
            "event": "sync_reconnect_failed", 
            "error": str(e),
            "error_type": type(e).__name__,
            "host": sync_host, 
            "port": sync_port
        })
        return None, None

def check_sync_socket(sock, sock_file, sync_host, sync_port):
    if sock is None:
        return connect_sync_socket(sync_host, sync_port)
    try:
        # Don't send ping, just check if usable by a dummy write (OS may throw if closed)
        sock.sendall(b"")  # lightweight check
        return sock, sock_file
    except Exception:
        return connect_sync_socket(sync_host, sync_port)

async def neural_agent_grpc(snake_id: str, log_file: str, env_host: str,
                model_save_dir: str = "models", learning_rate: float = 0.001,
                grpc_host: str = "localhost", grpc_port: int = 50051,
                batch_size: int = 5,
                sync_enabled: bool = False,
                sync_port: int = 5555,
                sync_buffer_size: int = 1024,
                           sync_host: str = "sync_service_host"):  # Add sync_host param
    """
    Neural agent with gRPC communication to training service.
    batch_size = number of episodes before sending to training
    Args:
        snake_id: ID of the snake
        log_file: Path to the log file
        env_host: Host of the snake game
        model_save_dir: Directory to save models
        learning_rate: Learning rate for optimizer
        grpc_host: gRPC training service host
        grpc_port: gRPC training service port
        batch_size: Number of episodes before sending to training
        sync_enabled: Whether to use TCP synchronization
        sync_port: Port to connect for sync signals
        sync_buffer_size: Buffer size for UDP socket
        sync_host: Host to connect for sync signals
    """
    base_url = f"http://{env_host}/snake/{snake_id}"
    move_url = f"{base_url}/move"
    reset_url = f"http://{env_host}/reset"
    
    logging.info({"event": "starting_grpc_neural_agent", "snake_id": snake_id})
    logging.info({"event": "batch_size_configured", "batch_size": batch_size, "unit": "episodes"})
    logging.info({"event": "grpc_training_service_configured", "host": grpc_host, "port": grpc_port})
    logging.info({"event": "mode_configured", "mode": "collect_episodes_then_batch_train"})
    
    # TCP Synchronization setup
    sync_socket = None
    sync_socket_file = None
    if sync_enabled:
        sync_socket, sync_socket_file = connect_sync_socket(sync_host, sync_port)

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
                    # Wait for sync signal if enabled
                    if sync_enabled:
                        sync_socket, sync_socket_file = check_sync_socket(sync_socket, sync_socket_file, sync_host, sync_port)
                        if sync_socket and sync_socket_file:
                            try:
                                # Detailed sync process logging
                                logging.debug({
                                    "event": "preparing_to_read_sync_signal",
                                    "socket_status": {
                                        "local_address": sync_socket.getsockname(),
                                        "remote_address": sync_socket.getpeername()
                                    }
                                })
                                
                                sync_socket.settimeout(15)  # Increased timeout

                                # Read initial message (could be a ready check)
                                line = sync_socket_file.readline()
                                # Log received line details
                                logging.debug({
                                    "event": "received_sync_signal_TCP", 
                                    "raw_line": line,
                                    "decoded_line": line.decode().strip() if line else None
                                })
                                
                                # Check if this is a ready check from server
                                decoded_line = line.decode().strip().lower() if line else ''
                                if decoded_line.startswith('sync_ready_check'):
                                    # Respond with ready message
                                    try:
                                        logging.debug({
                                            "event": "responding_to_sync_ready_check",
                                            "response": "ready"
                                        })
                                        sync_socket.sendall(b'ready\n')
                                    except Exception as send_error:
                                        logging.warning({
                                                    "event": "failed_to_send_ready",
                                                    "error": str(send_error)
                                        })
                                logging.info({
                                    "event": "sync_signal_processed",
                                    "status": "success",
                                    "server_message": decoded_line
                                })
                            except socket.timeout:
                                logging.warning({
                                    "event": "sync_signal_timeout", 
                                    "message": "Timeout waiting for sync signal"
                                })
                                sync_socket, sync_socket_file = None, None
                        continue

        except Exception as e:
                                logging.error({
                                    "event": "sync_signal_error_TCP", 
                                    "error": str(e),
                                    "error_type": type(e).__name__
                                })
                                sync_socket, sync_socket_file = None, None
                                continue
                        else:
                            logging.debug({
                                "event": "sync_socket_not_ready", 
                                "sync_socket": sync_socket is not None,
                                "sync_socket_file": sync_socket_file is not None
                            })

                    # Get latest state from stream
                    # non-blocking read (0.1 s timeout)
                    logging.info({"event": "get_latest_state"})
                    data, tech_timestamp = stream_reader.get_latest_state()
                    
                    if data is None or tech_timestamp is None:
                        logging.warning({"event": "no_data_received_from_stream"})
                        continue

                    send_datetime_str = data.get("datetime")
                    send_timestamp = datetime.fromisoformat(send_datetime_str).timestamp()
                    delay = tech_timestamp - send_timestamp
                    
                    logging.debug({"event": "state_received", "reward": data.get("reward", 0), "delay": f"{delay:.3f}s"})
                    
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
        logging.info({"event": "agent_interrupted_by_user"})
        
        # Send any remaining episodes before shutdown
        try:
            if len(agent.completed_episodes) > 0:
                logging.info({"event": "sending_final_batch", "episode_count": len(agent.completed_episodes)})
                await agent.send_training_batch_and_wait()
            saved_files = agent.save_all_data()
            logging.info({"event": "data_saved_on_shutdown", "saved_files": saved_files})
        except Exception as e:
            logging.error({"event": "error_during_shutdown_save", "exception": e})
            
    except Exception as e:
        logging.error({"event": "error_during_execution", "exception": e})
        raise
    finally:
        # Disconnect from training service
        await agent.disconnect_from_training_service()


def neural_agent(snake_id: str, log_file: str, env_host: str, 
                model_save_dir: str = "models", learning_rate: float = 0.001,
                grpc_host: str = "localhost", grpc_port: int = 50051,
                batch_size: int = 5,
                sync_enabled: bool = False,
                sync_port: int = 5555,
                sync_buffer_size: int = 1024,
                sync_host: str = "sync_service_host"):
    """
    Synchronous wrapper for running async gRPC agent.
    batch_size = number of episodes before sending to training
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
            batch_size=batch_size,
            sync_enabled=sync_enabled,
            sync_port=sync_port,
            sync_buffer_size=sync_buffer_size,
            sync_host=sync_host
        ))
    except KeyboardInterrupt:
        logging.info({"event": "agent_interrupted_by_user"})
    except Exception as e:
        logging.error({"event": "error_in_neural_agent", "exception": e})
        raise


def load_saved_model(model_path: str):
    """Function to load saved model in other code."""
    import torch
    from snake_model import ModelManager
    
    manager = ModelManager()
    model, optimizer, info = manager.load_model(model_path)
    return model, optimizer, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Snake Agent with gRPC Training (Episode-based Batching)")
    parser.add_argument("--snake_id", required=True, help="ID of the snake")
    parser.add_argument("--env_host", type=str, required=True, help="Host of the snake game")
    parser.add_argument("--log_file", default="neural_agent.log", help="Path to the log file")
    parser.add_argument("--model_save_dir", default="models", help="Directory to save models")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--grpc_host", default="localhost", help="gRPC training service host")
    parser.add_argument("--grpc_port", type=int, default=50051, help="gRPC training service port")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of episodes before sending to training")
    parser.add_argument("--sync_enabled", action="store_true", help="Enable TCP synchronization")
    parser.add_argument("--sync_port", type=int, default=5555, help="Port to connect for sync signals (TCP)")
    parser.add_argument("--sync_buffer_size", type=int, default=1024, help="Buffer size for UDP socket")
    parser.add_argument("--sync_host", type=str, default="sync_service_host", help="Host to connect for sync signals (TCP)")
    
    args = parser.parse_args()

    logger.setup_as_default()

    neural_agent(
        snake_id=args.snake_id,
        log_file=args.log_file,
        env_host=args.env_host,
        model_save_dir=args.model_save_dir, 
        learning_rate=args.learning_rate,
        grpc_host=args.grpc_host,
        grpc_port=args.grpc_port,
        batch_size=args.batch_size,
        sync_enabled=args.sync_enabled,
        sync_port=args.sync_port,
        sync_buffer_size=args.sync_buffer_size,
        sync_host=args.sync_host
    )
