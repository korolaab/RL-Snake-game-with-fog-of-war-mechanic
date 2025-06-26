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
                 grpc_host: str = "localhost", grpc_port: int = 50051, batch_size: int = 50):
        self.snake_id = snake_id
        self.model_save_dir = model_save_dir
        self.learning_rate = learning_rate
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.batch_size = batch_size
        
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
        
        # Experience buffer for batch transfer
        self.experience_buffer = []
        self.current_episode = 0
        self.batch_number = 0
        self.total_steps = 0
        
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
                logging.info(f"✅ Connected to training service at {self.grpc_host}:{self.grpc_port}")
            except asyncio.TimeoutError:
                raise ConnectionError("Timeout connecting to training service")
            
            # Start the model update stream
            await self._start_model_update_stream()
            
        except Exception as e:
            logging.error(f"❌ Failed to connect to training service: {e}")
            raise
    
    async def _start_model_update_stream(self):
        """Start the model update stream in the background."""
        try:
            logging.info("📡 Starting model update stream...")
            update_request = training_pb2.ModelUpdateRequest(snake_id=self.snake_id)
            
            # Start streaming in background task
            self.model_update_stream = asyncio.create_task(
                self._handle_model_update_stream(update_request)
            )
            
        except Exception as e:
            logging.error(f"Failed to start model update stream: {e}")
            raise
    
    async def _handle_model_update_stream(self, request):
        """Handle model updates from the stream."""
        try:
            async for model_update in self.training_stub.GetModelUpdates(request):
                if model_update.snake_id == self.snake_id:
                    logging.info(f"🔄 Received model update: step {model_update.training_step}")
                    # Put the update in the queue for processing
                    await self.pending_model_updates.put(model_update)
                    
        except grpc.RpcError as e:
            logging.error(f"Model update stream error: {e.code()}: {e.details()}")
        except Exception as e:
            logging.error(f"Model update stream error: {e}")
    
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
            logging.info("📡 Disconnected from training service")
    
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
                
                logging.info(f"✅ Loaded existing model: {model_info}")
                
                if model_info['snake_id'] != self.snake_id and model_info['snake_id'] != 'unknown':
                    logging.warning(f"Loaded model was trained for different snake_id: {model_info['snake_id']}")
            else:
                logging.info("🆕 No existing models found. Will create new model on first state (Cold Start).")
                self.is_cold_start = True
                
        except Exception as e:
            logging.error(f"Error loading existing model: {e}")
            logging.info("Will create new model on first state (Cold Start).")
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
            logging.info(f"🆕 Created fresh model for cold start, input_size: {input_size}")
        else:
            # Check size compatibility
            if not self.model_manager.validate_input_size(self.model, input_size):
                actual_size = self.model_manager.get_model_input_size(self.model)
                logging.warning(f"Input size mismatch. Expected: {actual_size}, Got: {input_size}")
                logging.info("Recreating model with correct input size...")
                
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
            logging.debug(f"Predicted action: {predicted_action} (probs: {action_probs.numpy()})")
            
            return predicted_action
            
        except Exception as e:
            logging.error(f"Error predicting action: {e}")
            # Return random action on error
            import random
            return random.choice(self.actions)
    
    def add_experience(self, state, action, reward, next_state=None, done=False):
        """Add experience to buffer."""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'step': self.total_steps
        }
        
        self.experience_buffer.append(experience)
        self.total_steps += 1
        
        logging.debug(f"Added experience: action={action}, reward={reward}, buffer_size={len(self.experience_buffer)}")
        
        # Check if we should send batch
        return len(self.experience_buffer) >= self.batch_size
    
    async def send_training_batch_and_wait(self):
        """Send training batch via gRPC and wait for model update."""
        if len(self.experience_buffer) == 0:
            logging.warning("No experiences to send")
            return False
        
        if not self.model_initialized:
            logging.warning("Model not initialized, cannot send batch")
            return False
        
        try:
            logging.info(f"📦 Sending training batch via gRPC: {len(self.experience_buffer)} experiences")
            
            # Convert experiences to protobuf format
            proto_experiences = []
            for exp in self.experience_buffer:
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
                model_data = pickle.dumps(self.model)
                logging.debug(f"Serialized model, size: {len(model_data)} bytes")
            except Exception as e:
                logging.error(f"Failed to serialize model: {e}")
                return False
            
            # Create training batch request
            request = training_pb2.TrainingBatchRequest(
                snake_id=self.snake_id,
                timestamp=datetime.now().isoformat(),
                episode=self.current_episode,
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
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logging.error("❌ Timeout sending training batch")
                return False
            except grpc.RpcError as e:
                logging.error(f"❌ gRPC error sending batch: {e.code()}: {e.details()}")
                return False
            
            if response.success:
                logging.info(f"✅ Training batch sent successfully: {response.message}")
                
                # Wait for model update from stream
                logging.info("⏳ Waiting for model update...")
                
                try:
                    # Wait for model update with timeout
                    model_update = await asyncio.wait_for(
                        self.pending_model_updates.get(),
                        timeout=1.0
                    )
                    
                    logging.info(f"🔄 Processing model update: step {model_update.training_step}")
                    
                    # Deserialize updated model
                    try:
                        updated_model = pickle.loads(model_update.model_data)
                        if not isinstance(updated_model, torch.nn.Module):
                            raise ValueError(f"Received invalid model type: {type(updated_model)}")
                        
                        # Update our model
                        self.model = updated_model
                        self.model.train()  # Ensure training mode
                        
                        # Update optimizer with new model parameters
                        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                        
                        logging.info("✅ Model updated successfully!")
                        
                        # Log training metrics
                        metrics = model_update.metrics
                        logging.info(f"📊 Training metrics - Loss: {metrics.loss:.4f}, "
                                   f"Avg Reward: {metrics.avg_reward:.4f}, "
                                   f"Samples: {metrics.num_samples}")
                        
                    except Exception as e:
                        logging.error(f"Failed to deserialize updated model: {e}")
                        return False
                    
                except asyncio.TimeoutError:
                    logging.error("❌ Timeout waiting for model update")
                    return False
                except Exception as e:
                    logging.error(f"❌ Error getting model update: {e}")
                    return False
                
                # Clear buffer and update counters
                self.experience_buffer = []
                self.batch_number += 1
                
                # After first batch, no longer cold start
                if self.is_cold_start:
                    self.is_cold_start = False
                    logging.info("🚀 Cold start completed")
                
                return True
            else:
                logging.error(f"❌ Training batch failed: {response.message}")
                return False
                
        except Exception as e:
            logging.error(f"Error sending training batch via gRPC: {e}")
            return False
    
    def save_experience(self, state, reward, action):
        """Backward compatibility - add experience to buffer."""
        return self.add_experience(state, action, reward)
    
    def save_all_data(self):
        """Save all episode data."""
        try:
            saved_files = {}
            
            # Save model (backup)
            if self.model is not None:
                model_path = self.model_manager.save_model(
                    self.model, self.optimizer, self.snake_id
                )
                saved_files['model'] = model_path
            
            # Save history and weights (local backup)
            data_files = self.data_manager.save_all_data(self.model, self.snake_id)
            saved_files.update(data_files)
            
            # Complete episode
            self.current_episode += 1
            
            # Output statistics
            stats = self.data_manager.get_statistics()
            logging.info(f"📊 Episode {self.current_episode} completed. Statistics: {stats}")
            
            return saved_files
            
        except Exception as e:
            logging.error(f"Error saving data: {e}")
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
            'experience_buffer_size': len(self.experience_buffer),
            'current_episode': self.current_episode,
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
        return self.data_manager.get_statistics()
