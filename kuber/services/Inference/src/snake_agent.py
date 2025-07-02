import torch
import logging
import asyncio
import grpc
import json
import pickle
from datetime import datetime
from snake_model import ModelManager
from state_processor import StateProcessor
from data_manager import DataManager
import io

# Import generated protobuf classes
try:
    import training_pb2
    import training_pb2_grpc
except ImportError:
    print("❌ Error: Generated protobuf files not found!")
    print("Please run generate_grpc.py first to generate training_pb2.py and training_pb2_grpc.py")
    raise

class GRPCSnakeAgent:
    """Snake Agent with gRPC communication to training service."""
    
    def __init__(self, snake_id: str, model_save_dir: str = "models", learning_rate: float = 0.001,
                 grpc_host: str = "localhost", grpc_port: int = 50051, batch_size: int = 5):
        self.snake_id = snake_id
        self.model_save_dir = model_save_dir
        self.learning_rate = learning_rate
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.batch_size = batch_size  # Number of EPISODES before sending batch
        
        # Initialize components
        self.model_manager = ModelManager(model_save_dir)
        self.state_processor = StateProcessor()
        self.data_manager = DataManager(model_save_dir)
        
        # Model and optimizer
        self.model = None
        self.optimizer = None
        self.model_initialized = False
        self.model_info = None
        self.is_cold_start = False
        
        # Episode and experience tracking
        self.completed_episodes = []  # List of completed episodes
        self.current_episode_experiences = []  # Current episode's experiences
        self.current_episode = 0
        self.batch_number = 0
        self.total_steps = 0
        self.current_episode_steps = 0
        
        # Actions
        self.actions = ["left", "right", "forward"]
        
        # gRPC client
        self.channel = None
        self.training_stub = None
        
        # Model update tracking
        self.pending_model_updates = asyncio.Queue()
        self.model_update_stream = None
        
        # Load existing model
        self.load_existing_model()
    
    async def connect_to_training_service(self):
        """Connect to the gRPC training service and start model update stream."""
        try:
            self.channel = grpc.aio.insecure_channel(f'{self.grpc_host}:{self.grpc_port}')
            self.training_stub = training_pb2_grpc.TrainingServiceStub(self.channel)
            
            # Test connection with a timeout
            try:
                await asyncio.wait_for(
                    self.channel.channel_ready(), 
                    timeout=10.0
                )
                logging.info({"event": "connected_to_training_service", "host": self.grpc_host, "port": self.grpc_port})
            except asyncio.TimeoutError:
                raise ConnectionError("Timeout connecting to training service")
            
            # Start the model update stream
            await self._start_model_update_stream()
            
        except Exception as e:
            logging.error({"event": "failed_to_connect_to_training_service", "exception": e})
            raise
    
    async def _start_model_update_stream(self):
        """Start the model update stream in the background."""
        try:
            logging.info({"event": "starting_model_update_stream"})
            update_request = training_pb2.ModelUpdateRequest(snake_id=self.snake_id)
            
            # Start streaming in background task
            self.model_update_stream = asyncio.create_task(
                self._handle_model_update_stream(update_request)
            )
            
        except Exception as e:
            logging.error({"event": "failed_to_start_model_update_stream", "exception": e})
            raise
    
    async def _handle_model_update_stream(self, request):
        """Handle model updates from the stream."""
        try:
            async for model_update in self.training_stub.GetModelUpdates(request):
                if model_update.snake_id == self.snake_id:
                    logging.info({"event": "received_model_update", "training_step": model_update.training_step})
                    # Put the update in the queue for processing
                    await self.pending_model_updates.put(model_update)
                    
        except grpc.RpcError as e:
            logging.error({"event": "model_update_stream_error", "error_code": e.code(), "error_details": e.details()})
        except Exception as e:
            logging.error({"event": "model_update_stream_error", "exception": e})
    
    async def disconnect_from_training_service(self):
        """Disconnect from the gRPC training service."""
        if self.model_update_stream:
            self.model_update_stream.cancel()
            try:
                await self.model_update_stream
            except asyncio.CancelledError:
                pass
        
        if self.channel:
            await self.channel.close()
            logging.info({"event": "disconnected_from_training_service"})
    
    def load_existing_model(self):
        """Load existing model at initialization."""
        try:
            model, optimizer, model_info = self.model_manager.load_latest_model(
                self.snake_id, self.learning_rate
            )
            
            if model is not None:
                self.model = model
                self.optimizer = optimizer
                self.model_info = model_info
                self.model_initialized = True
                self.is_cold_start = False
                
                logging.info({"event": "loaded_existing_model", "model_info": model_info})
                
                if model_info['snake_id'] != self.snake_id and model_info['snake_id'] != 'unknown':
                    logging.warning({"event": "loaded_model_different_snake_id", "model_snake_id": model_info["snake_id"]})
            else:
                logging.info({"event": "no_existing_models_found", "action": "will_create_new_model_on_cold_start"})
                self.is_cold_start = True
                
        except Exception as e:
            logging.error({"event": "error_loading_existing_model", "exception": e})
            logging.info({"event": "will_create_new_model_on_cold_start"})
            self.is_cold_start = True
    
    def ensure_model_initialized(self, input_size: int):
        """Ensure model is initialized with correct dimensions."""
        if not self.model_initialized:
            # Cold start - create new model
            self.model, self.optimizer = self.model_manager.create_new_model(
                input_size, self.learning_rate
            )
            self.model_initialized = True
            self.is_cold_start = True
            logging.info({"event": "created_fresh_model_for_cold_start", "input_size": input_size})
        else:
            # Check size compatibility
            if not self.model_manager.validate_input_size(self.model, input_size):
                actual_size = self.model_manager.get_model_input_size(self.model)
                logging.warning({"event": "input_size_mismatch", "expected": actual_size, "got": input_size})
                logging.info({"event": "recreating_model_with_correct_input_size"})
                
                self.model, self.optimizer = self.model_manager.create_new_model(
                    input_size, self.learning_rate
                )
                self.is_cold_start = True
    
    def predict_action(self, state):
        """Predict action based on state."""
        try:
            # Process state
            state_tensor = self.state_processor.process_state(state)
            
            # Convert to flat vector for model input
            flat_tensor = state_tensor.flatten().unsqueeze(0)  # add batch dimension
            input_size = flat_tensor.shape[1]
            
            # Ensure model is initialized
            self.ensure_model_initialized(input_size)
            
            # Prediction
            with torch.no_grad():
                self.model.eval()  # Set to evaluation mode
                action_probs = self.model(flat_tensor)
                m = torch.distributions.Categorical(action_probs)
                action_idx = m.sample()
                self.model.train()  # Set back to training mode

            predicted_action = self.actions[action_idx]
            logging.debug({"event": "predicted_action", "action": predicted_action, "probabilities": action_probs.numpy()})
            
            return predicted_action
            
        except Exception as e:
            logging.error({"event": "error_predicting_action", "exception": e})
            # Return random action on error
            import random
            return random.choice(self.actions)
    
    def add_experience(self, state, action, reward, next_state=None, done=False):
        """Add experience to current episode."""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'step': self.current_episode_steps
        }
        
        self.current_episode_experiences.append(experience)
        self.total_steps += 1
        self.current_episode_steps += 1
        
        logging.debug({"event": "added_experience", "action": action, "reward": reward, "episode_steps": self.current_episode_steps})
        
        # If episode is done, complete the episode
        if done:
            return self._complete_episode()
        
        return False  # Don't send batch until episode is complete
    
    def _complete_episode(self):
        """Complete the current episode and check if we should send a batch."""
        if not self.current_episode_experiences:
            logging.warning({"event": "no_experiences_in_current_episode"})
            return False
        
        # Create episode summary
        episode_data = {
            'episode_number': self.current_episode,
            'experiences': self.current_episode_experiences.copy(),
            'total_steps': len(self.current_episode_experiences),
            'total_reward': sum(exp['reward'] for exp in self.current_episode_experiences),
            'final_reward': self.current_episode_experiences[-1]['reward'] if self.current_episode_experiences else 0
        }
        
        # Add completed episode to batch
        self.completed_episodes.append(episode_data)
        
        logging.info(f"🎮 Episode {self.current_episode} completed: {len(self.current_episode_experiences)} steps, "
                    f"total_reward={episode_data['total_reward']}, final_reward={episode_data['final_reward']}")
        
        # Reset for next episode
        self.current_episode_experiences = []
        self.current_episode += 1
        self.current_episode_steps = 0
        
        # Check if we should send batch (have enough completed episodes)
        should_send_batch = len(self.completed_episodes) >= self.batch_size
        
        if should_send_batch:
            logging.info({"event": "ready_to_send_batch", "completed_episodes": len(self.completed_episodes)})
        
        return should_send_batch
    
    async def send_training_batch_and_wait(self):
        """Send training batch via gRPC and wait for model update."""
        if len(self.completed_episodes) == 0:
            logging.warning({"event": "no_completed_episodes_to_send"})
            return False
        
        if not self.model_initialized:
            logging.warning({"event": "model_not_initialized", "action": "cannot_send_batch"})
            return False
        
        try:
            # Calculate total experiences across all episodes
            total_experiences = sum(len(ep['experiences']) for ep in self.completed_episodes)
            total_episodes = len(self.completed_episodes)
            
            logging.info({"event": "sending_training_batch_via_grpc", "total_episodes": total_episodes, "total_experiences": total_experiences})
            
            # Convert all experiences to protobuf format
            proto_experiences = []
            for episode in self.completed_episodes:
                for exp in episode['experiences']:
                    proto_exp = training_pb2.Experience(
                        state_json=json.dumps(exp['state']),
                        action=exp['action'],
                        reward=exp['reward'],
                        step=exp['step'],
                        done=exp['done']
                    )
                    proto_experiences.append(proto_exp)
            
            # Serialize model with error handling
            try:
                scripted_model = torch.jit.script(self.model)
                buffer = io.BytesIO()
                torch.jit.save(scripted_model, buffer)
                model_data = buffer.getvalue()
                logging.debug({"event": "serialized_model", "size_bytes": len(model_data)})
            except Exception as e:
                logging.error({"event": "failed_to_serialize_model", "exception": e})
                return False
            
            # Create training batch request
            request = training_pb2.TrainingBatchRequest(
                snake_id=self.snake_id,
                timestamp=datetime.now().isoformat(),
                episode=self.current_episode,  # Next episode number
                batch_number=self.batch_number,
                total_steps=self.total_steps,
                is_cold_start=self.is_cold_start,
                model_data=model_data,
                experiences=proto_experiences
            )
            
            # Send training batch with timeout
            try:
                response = await asyncio.wait_for(
                    self.training_stub.SendTrainingBatch(request),
                    timeout=60.0  # Longer timeout for larger batches
                )
            except asyncio.TimeoutError:
                logging.error({"event": "timeout_sending_training_batch"})
                return False
            except grpc.RpcError as e:
                logging.error({"event": "grpc_error_sending_batch", "error_code": e.code(), "error_details": e.details()})
                return False
            
            if response.success:
                await asyncio.sleep(0.1) #TODO This is a workarround for race condition and time sync problem. Training can send result earlyer than inference ready to read it
                logging.info({"event": "training_batch_sent_successfully", "response_message": response.message})
                
                # Wait for model update from stream
                logging.info({"event": "waiting_for_model_update"})
                
                try:
                    # Wait for model update with timeout
                    model_update = await asyncio.wait_for(
                        self.pending_model_updates.get(),
                        timeout=60.0  # Longer timeout for training
                    )
                    
                    logging.info({"event": "processing_model_update", "training_step": model_update.training_step})
                    
                    # Deserialize updated model
                    try:
                        buffer = io.BytesIO(model_update.model_data)
                        updated_model = torch.jit.load(buffer, map_location='cpu')
                        if not isinstance(updated_model, torch.nn.Module):
                            raise ValueError(f"Received invalid model type: {type(updated_model)}")
                        
                        # Update our model
                        self.model = updated_model
                        self.model.train()  # Ensure training mode
                        
                        # Update optimizer with new model parameters
                        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                        
                        logging.info({"event": "model_updated_successfully"})
                        
                        # Log training metrics
                        metrics = model_update.metrics
                        logging.info(f"📊 Training metrics - Loss: {metrics.loss:.4f}, "
                                   f"Avg Reward: {metrics.avg_reward:.4f}, "
                                   f"Episodes: {total_episodes}, Total Experiences: {metrics.num_samples}")
                        
                    except Exception as e:
                        logging.error({"event": "failed_to_deserialize_updated_model", "exception": e})
                        return False
                    
                except asyncio.TimeoutError:
                    logging.error({"event": "timeout_waiting_for_model_update"})
                    return False
                except Exception as e:
                    logging.error({"event": "error_getting_model_update", "exception": e})
                    return False
                
                # Clear completed episodes and update counters
                self.completed_episodes = []
                self.batch_number += 1
                
                # After first batch, no longer cold start
                if self.is_cold_start:
                    self.is_cold_start = False
                    logging.info({"event": "cold_start_completed"})
                
                return True
            else:
                logging.error({"event": "training_batch_failed", "response_message": response.message})
                return False
                
        except Exception as e:
            logging.error({"event": "error_sending_training_batch_via_grpc", "exception": e})
            return False
    
    def save_experience(self, state, reward, action):
        """Backward compatibility - add experience to current episode."""
        return self.add_experience(state, action, reward)
    
    def save_all_data(self):
        """Save all episode data and send final batch if needed."""
        try:
            saved_files = {}
            
            # If we have a partial current episode, complete it
            if self.current_episode_experiences:
                logging.info({"event": "completing_partial_episode", "episode": self.current_episode, "experience_count": len(self.current_episode_experiences)})
                # Mark last experience as done to complete episode
                if self.current_episode_experiences:
                    self.current_episode_experiences[-1]['done'] = True
                self._complete_episode()
            
            # Save model (backup)
            if self.model is not None:
                model_path = self.model_manager.save_model(
                    self.model, self.optimizer, self.snake_id
                )
                saved_files['model'] = model_path
            
            # Save history and weights (local backup)
            data_files = self.data_manager.save_all_data(self.model, self.snake_id)
            saved_files.update(data_files)
            
            # Output statistics
            stats = self.data_manager.get_statistics()
            logging.info({"event": "game_session_completed", "statistics": stats})
            logging.info({"event": "total_episodes_completed", "count": self.current_episode})
            logging.info({"event": "total_steps_across_all_episodes", "count": self.total_steps})
            
            return saved_files
            
        except Exception as e:
            logging.error({"event": "error_saving_data", "exception": e})
            raise
    
    def get_model_info(self):
        """Get current model information."""
        info = {
            'model_initialized': self.model_initialized,
            'snake_id': self.snake_id,
            'actions': self.actions,
            'learning_rate': self.learning_rate,
            'is_cold_start': self.is_cold_start,
            'batch_size': self.batch_size,
            'completed_episodes': len(self.completed_episodes),
            'current_episode': self.current_episode,
            'current_episode_steps': len(self.current_episode_experiences),
            'batch_number': self.batch_number,
            'total_steps': self.total_steps,
            'grpc_host': self.grpc_host,
            'grpc_port': self.grpc_port
        }
        
        if self.model is not None:
            info['input_size'] = self.model_manager.get_model_input_size(self.model)
        
        if self.model_info is not None:
            info['loaded_from'] = self.model_info
        
        return info
    
    def get_statistics(self):
        """Get accumulated data statistics."""
        stats = self.data_manager.get_statistics()
        stats.update({
            'episodes_completed': self.current_episode,
            'episodes_in_current_batch': len(self.completed_episodes),
            'current_episode_steps': len(self.current_episode_experiences),
            'total_steps': self.total_steps
        })
        return stats
