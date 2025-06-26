import asyncio
import grpc
import pickle
import json
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import argparse
import traceback
from concurrent import futures

# Import generated protobuf classes
try:
    import training_pb2
    import training_pb2_grpc
except ImportError:
    print("❌ Error: Generated protobuf files not found!")
    print("Please run generate_grpc.py first to generate training_pb2.py and training_pb2_grpc.py")
    sys.exit(1)

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

class SimpleModel(nn.Module):
    """Simple model for creating from scratch if needed."""
    def __init__(self, input_size, hidden_size=64, output_size=3):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class GRPCTrainingService(training_pb2_grpc.TrainingServiceServicer):
    """gRPC Training Service Implementation."""
    
    def __init__(self, snake_id: str, learning_rate: float = 0.001):
        self.snake_id = snake_id
        self.learning_rate = learning_rate
        self.model = None
        self.optimizer = None
        self.training_step = 0
        
        # Store model updates for streaming - one queue per snake
        self.model_update_queues = {}
        
    def setup_model_and_optimizer(self, model, is_cold_start: bool):
        """Setup model and optimizer from received model."""
        # Ensure model is in training mode
        self.model = model
        self.model.train()
        
        # Create fresh optimizer for this model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if is_cold_start:
            logging.info("🆕 Received fresh model from cold start inference")
        else:
            logging.info("🔄 Received existing model from inference")
    
    def process_experiences_for_training(self, experiences: List[training_pb2.Experience]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process experience data for training."""
        states = []
        rewards = []
        actions = []
        
        for exp in experiences:
            # Parse state from JSON
            try:
                state = json.loads(exp.state_json)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse state JSON: {e}")
                continue
                
            reward = exp.reward
            action = exp.action
            
            # Extract visible cells and process
            visible_cells = state.get('visible_cells', {})
            filtered_cells = {k: v for k, v in visible_cells.items() if v != 'HEAD'}
            
            if not filtered_cells:
                continue
                
            # Sort cells by coordinates
            sorted_cells = []
            for coord_str, cell_type in filtered_cells.items():
                try:
                    x, y = map(int, coord_str.split(','))
                    sorted_cells.append((x, y, cell_type))
                except ValueError:
                    continue
            
            if not sorted_cells:
                continue
                
            sorted_cells.sort(key=lambda item: (item[0], item[1]))
            
            # Encode cells
            cell_encoding = {
                'FOOD': [0, 0, 1],
                'BODY': [0, 1, 0],
                'OTHER_BODY': [1, 0, 0],
                'EMPTY': [0, 0, 0]
            }
            
            tensor_cell_data = []
            for x, y, cell_type in sorted_cells:
                encoding = cell_encoding.get(cell_type, [0, 0, 0])
                tensor_cell_data.append(encoding)

            # Encode actions
            action_encoding = {
                'forward': [0, 0, 1],
                'right': [0, 1, 0],
                'left': [1, 0, 0]
            }
            
            action_vector = action_encoding.get(action, [0, 0, 1])

            rewards.append(reward)

            state_tensor = torch.tensor(tensor_cell_data, dtype=torch.float32).flatten()
            states.append(state_tensor)

            action_tensor = torch.tensor(action_vector, dtype=torch.float32)
            actions.append(action_tensor)
        
        if not states:
            raise ValueError("No valid states found in experiences")
        
        # Pad states to same length
        max_length = max(len(state) for state in states)
        padded_states = []
        
        for state in states:
            if len(state) < max_length:
                padded_state = torch.zeros(max_length)
                padded_state[:len(state)] = state
                padded_states.append(padded_state)
            else:
                padded_states.append(state)
        
        states_tensor = torch.stack(padded_states)
        actions_tensor = torch.stack(actions)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        logging.info(f"Processed {len(states)} states, tensor shape: {states_tensor.shape}")
        return states_tensor, rewards_tensor, actions_tensor
    
    def perform_training_step(self, states: torch.Tensor, actions_probs: torch.Tensor, rewards: torch.Tensor) -> Dict[str, float]:
        """Perform one training step."""
        logging.info(f"🏋️ Training step {self.training_step + 1} starting...")
        
        if self.model is None:
            # Create fallback model
            self.model = SimpleModel(states.shape[1])
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            logging.info(f"Created fallback model with input_size: {states.shape[1]}")
        
        try:
            actions = torch.argmax(actions_probs, dim=1)
            
            # Forward pass
            probs = self.model(states)
            m = torch.distributions.Categorical(probs)
            log_probs = m.log_prob(actions)
            entropy = m.entropy()
            
            # Calculate discounted returns
            gamma = 0.9
            ep_returns = torch.zeros_like(rewards)
            ep_returns[-1] = rewards[-1]
            for t in reversed(range(len(rewards) - 1)):
                ep_returns[t] = rewards[t] + gamma * ep_returns[t + 1]

            # Normalize returns
            if len(ep_returns) > 1:
                ep_returns = (ep_returns - ep_returns.mean()) / (ep_returns.std() + 1e-5)
            
            # Calculate loss
            beta = 0.2
            loss = -(log_probs * ep_returns).sum() - beta * entropy.sum()
            
            if torch.isnan(loss):
                raise ValueError("Loss is NaN!")

            # Perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                avg_reward = rewards.mean()
                max_action_prob = actions_probs.max(dim=-1)[0].mean()
                
                metrics = {
                    'loss': loss.item(),
                    'entropy_mean': entropy.mean().item(),
                    'avg_reward': avg_reward.item(),
                    'max_action_prob': max_action_prob.item(),
                    'num_samples': len(states),
                    'positive_rewards': (rewards > 0).sum().item(),
                    'negative_rewards': (rewards <= 0).sum().item()
                }
            
            self.training_step += 1
            logging.info(f"✅ Training step {self.training_step} completed! Loss: {loss.item():.4f}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error during training step: {e}")
            raise
    
    async def SendTrainingBatch(self, request: training_pb2.TrainingBatchRequest, context) -> training_pb2.TrainingBatchResponse:
        """Handle training batch request."""
        try:
            logging.info(f"📦 Received training batch: {len(request.experiences)} experiences")
            
            # Deserialize model with error handling
            try:
                model = pickle.loads(request.model_data)
                if not isinstance(model, torch.nn.Module):
                    raise ValueError(f"Deserialized object is not a PyTorch model: {type(model)}")
            except Exception as e:
                logging.error(f"Failed to deserialize model: {e}")
                return training_pb2.TrainingBatchResponse(
                    success=False,
                    message=f"Model deserialization failed: {str(e)}",
                    training_step=self.training_step
                )
            
            # Setup model and optimizer
            self.setup_model_and_optimizer(model, request.is_cold_start)
            
            # Process experiences for training
            try:
                states, rewards, actions = self.process_experiences_for_training(request.experiences)
            except Exception as e:
                logging.error(f"Failed to process experiences: {e}")
                return training_pb2.TrainingBatchResponse(
                    success=False,
                    message=f"Experience processing failed: {str(e)}",
                    training_step=self.training_step
                )
            
            # Train the model
            try:
                metrics = self.perform_training_step(states, actions, rewards)
            except Exception as e:
                logging.error(f"Training step failed: {e}")
                return training_pb2.TrainingBatchResponse(
                    success=False,
                    message=f"Training failed: {str(e)}",
                    training_step=self.training_step
                )
            
            # Create model update
            try:
                # Serialize model with proper error handling
                model_data = pickle.dumps(self.model)
                
                model_update = training_pb2.ModelUpdateResponse(
                    snake_id=request.snake_id,
                    timestamp=datetime.now().isoformat(),
                    training_step=self.training_step,
                    model_data=model_data,
                    metrics=training_pb2.TrainingMetrics(
                        loss=metrics['loss'],
                        entropy_mean=metrics['entropy_mean'],
                        avg_reward=metrics['avg_reward'],
                        max_action_prob=metrics['max_action_prob'],
                        num_samples=metrics['num_samples'],
                        positive_rewards=metrics['positive_rewards'],
                        negative_rewards=metrics['negative_rewards']
                    )
                )
                
                # Send update to all queues for this snake_id
                if request.snake_id in self.model_update_queues:
                    for queue in self.model_update_queues[request.snake_id]:
                        try:
                            await asyncio.wait_for(queue.put(model_update), timeout=1.0)
                            logging.info(f"📤 Queued model update for {request.snake_id}: step {self.training_step}")
                        except asyncio.TimeoutError:
                            logging.warning(f"Timeout queuing model update for {request.snake_id}")
                        except Exception as e:
                            logging.warning(f"Error queuing model update: {e}")
                
                logging.info(f"📊 Training metrics: {metrics}")
                
                return training_pb2.TrainingBatchResponse(
                    success=True,
                    message="Training completed successfully",
                    training_step=self.training_step
                )
                
            except Exception as e:
                logging.error(f"Failed to create model update: {e}")
                return training_pb2.TrainingBatchResponse(
                    success=False,
                    message=f"Model update creation failed: {str(e)}",
                    training_step=self.training_step
                )
            
        except Exception as e:
            logging.error(f"Error processing training batch: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return training_pb2.TrainingBatchResponse(
                success=False,
                message=f"Training failed: {str(e)}",
                training_step=self.training_step
            )
    
    async def GetModelUpdates(self, request: training_pb2.ModelUpdateRequest, context):
        """Stream model updates to inference."""
        logging.info(f"📡 Starting model update stream for {request.snake_id}")
        
        # Create a queue for this specific stream
        update_queue = asyncio.Queue()
        
        # Add queue to the snake's queue list
        if request.snake_id not in self.model_update_queues:
            self.model_update_queues[request.snake_id] = []
        self.model_update_queues[request.snake_id].append(update_queue)
        
        try:
            while True:
                # Wait for model updates
                try:
                    model_update = await update_queue.get()
                    logging.info(f"📤 Streaming model update: step {model_update.training_step}")
                    yield model_update
                except Exception as e:
                    logging.error(f"Error getting model update from queue: {e}")
                    break
                
        except asyncio.CancelledError:
            logging.info(f"Model update stream cancelled for {request.snake_id}")
        except Exception as e:
            logging.error(f"Error in model update stream: {e}")
        finally:
            # Clean up: remove this queue from the snake's queue list
            if request.snake_id in self.model_update_queues:
                try:
                    self.model_update_queues[request.snake_id].remove(update_queue)
                    if not self.model_update_queues[request.snake_id]:
                        del self.model_update_queues[request.snake_id]
                except ValueError:
                    pass  # Queue already removed
            logging.info(f"📡 Model update stream ended for {request.snake_id}")

async def serve(port: int, snake_id: str, learning_rate: float):
    """Start the gRPC server."""
    # Create the training service
    training_service = GRPCTrainingService(snake_id, learning_rate)
    
    # Create server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add the service
    training_pb2_grpc.add_TrainingServiceServicer_to_server(training_service, server)
    
    logging.info(f"🚀 Starting gRPC training server on port {port}")
    
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    
    await server.start()
    logging.info(f"✅ gRPC server started and listening on {listen_addr}")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("🛑 Shutting down gRPC server...")
        await server.stop(5)

async def main():
    parser = argparse.ArgumentParser(description="gRPC Training Server")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--snake_id", required=True, help="Snake ID")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--log_file", default="grpc_training.log", help="Log file path")
    
    args = parser.parse_args()
    
    setup_logger(args.log_file)
    
    await serve(args.port, args.snake_id, args.learning_rate)

if __name__ == "__main__":
    asyncio.run(main())
