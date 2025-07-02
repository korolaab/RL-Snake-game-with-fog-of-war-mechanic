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
import logging
import logger
import io

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
            logging.info({"event": "received_fresh_model", "source": "cold_start_inference"})
        else:
            logging.info({"event": "received_existing_model", "source": "inference"})
    
    def organize_experiences_by_episodes(self, experiences: List[training_pb2.Experience]) -> List[List[Dict]]:
        """Organize experiences into separate episodes based on 'done' flag."""
        episodes = []
        current_episode = []
        
        for exp in experiences:
            # Parse state from JSON
            try:
                state = json.loads(exp.state_json)
            except json.JSONDecodeError as e:
                logging.warning({"event": "failed_to_parse_state_json", "exception": e})
                continue
            
            experience_dict = {
                'state': state,
                'action': exp.action,
                'reward': exp.reward,
                'step': exp.step,
                'done': exp.done
            }
            
            current_episode.append(experience_dict)
            
            # If episode is done, start a new episode
            if exp.done:
                if current_episode:  # Only add non-empty episodes
                    episodes.append(current_episode)
                    current_episode = []
        
        # Add any remaining experiences as the last episode (shouldn't happen but defensive)
        if current_episode:
            logging.warning({"event": "found_incomplete_episode", "experience_count": len(current_episode)})
            episodes.append(current_episode)
        
        logging.info({"event": "organized_experiences", "experience_count": len(experiences), "episode_count": len(episodes)})
        for i, episode in enumerate(episodes):
            total_reward = sum(exp['reward'] for exp in episode)
            logging.info({"event": "episode_summary", "episode_num": i+1, "step_count": len(episode), "total_reward": total_reward})
        
        return episodes
    
    def process_episodes_for_training(self, episodes: List[List[Dict]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process episode data for REINFORCE training with proper returns calculation."""
        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        
        for episode_idx, episode in enumerate(episodes):
            if not episode:
                continue
                
            episode_states = []
            episode_actions = []
            episode_rewards = []
            
            # Process each step in the episode
            for exp in episode:
                state = exp['state']
                action = exp['action']
                reward = exp['reward']
                
                # Extract and encode visible cells
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
                    'forward': 0,
                    'right': 1,
                    'left': 2
                }
                
                action_idx = action_encoding.get(action, 0)

                # Store processed data
                state_tensor = torch.tensor(tensor_cell_data, dtype=torch.float32).flatten()
                episode_states.append(state_tensor)
                episode_actions.append(action_idx)
                episode_rewards.append(reward)
            
            if not episode_states:
                logging.warning({"event": "no_valid_states_in_episode", "episode_idx": episode_idx})
                continue
            
            # Calculate discounted returns for this episode (REINFORCE style)
            gamma = 0.99
            episode_returns = []
            discounted_return = 0
            
            # Calculate returns from end to beginning (proper REINFORCE)
            for reward in reversed(episode_rewards):
                discounted_return = reward + gamma * discounted_return
                episode_returns.insert(0, discounted_return)
            
            # Convert to tensors
            episode_returns = torch.tensor(episode_returns, dtype=torch.float32)
            
            # Normalize returns within episode (reduce variance)
            if len(episode_returns) > 1:
                episode_returns = (episode_returns - episode_returns.mean()) / (episode_returns.std() + 1e-8)
            
            # Add to global lists
            all_states.extend(episode_states)
            all_actions.extend(episode_actions)
            all_returns.extend(episode_returns.tolist())
            
            logging.info({"event": f"Episode {episode_idx+1} processed", 
                          "episode_states_len": len(episode_states)})
        
        if not all_states:
            raise ValueError("No valid states found across all episodes")
        
        # Pad states to same length
        max_length = max(len(state) for state in all_states)
        padded_states = []
        
        for state in all_states:
            if len(state) < max_length:
                padded_state = torch.zeros(max_length)
                padded_state[:len(state)] = state
                padded_states.append(padded_state)
            else:
                padded_states.append(state)
        
        # Convert to tensors
        states_tensor = torch.stack(padded_states)
        actions_tensor = torch.tensor(all_actions, dtype=torch.long)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
        
        logging.info({"event": "final_batch_prepared", "transition_count": len(all_states), "episode_count": len(episodes)})
        logging.info({"event": "tensor_shape_info", "tensor_type": "states", "shape": states_tensor.shape})
        logging.info({"event": "tensor_shape_info", "tensor_type": "actions", "shape": actions_tensor.shape})
        logging.info({"event": "tensor_shape_info", "tensor_type": "returns", "shape": returns_tensor.shape})
        logging.info({"event": "return_statistics", "mean": f"{returns_tensor.mean():.3f}", "std": f"{returns_tensor.std():.3f}"})
        
        return states_tensor, actions_tensor, returns_tensor, max_length
    
    def perform_training_step(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor, input_size: int) -> Dict[str, float]:
        """Perform one REINFORCE training step on episode batch."""
        logging.info({"event": "training_step_started", "step": self.training_step + 1})
        
        if self.model is None:
            # Create fallback model
            self.model = SimpleModel(input_size)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            logging.info({"event": "created_fallback_model", "input_size": input_size})
        
        try:
            # Forward pass through policy network
            action_probs = self.model(states)  # Shape: (batch_size, num_actions)
            
            # Calculate log probabilities for taken actions
            m = torch.distributions.Categorical(action_probs)
            log_probs = m.log_prob(actions)  # Shape: (batch_size,)
            entropy = m.entropy().mean()  # Average entropy for regularization
            
            # REINFORCE loss: -log(π(a|s)) * R
            # Negative because we want to maximize, but optimizers minimize
            policy_loss = -(log_probs * returns).mean()
            
            # Add entropy bonus to encourage exploration
            entropy_coeff = 0.01
            total_loss = policy_loss - entropy_coeff * entropy
            
            if torch.isnan(total_loss):
                raise ValueError("Loss is NaN!")

            # Perform backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                avg_return = returns.mean()
                max_action_prob = action_probs.max(dim=-1)[0].mean()
                
                # Calculate action distribution
                action_counts = torch.bincount(actions, minlength=3)
                action_distribution = action_counts.float() / len(actions)
                
                metrics = {
                    'loss': total_loss.item(),
                    'policy_loss': policy_loss.item(),
                    'entropy_mean': entropy.item(),
                    'avg_return': avg_return.item(),
                    'max_action_prob': max_action_prob.item(),
                    'num_samples': len(states),
                    'positive_returns': (returns > 0).sum().item(),
                    'negative_returns': (returns <= 0).sum().item(),
                    'action_dist_left': action_distribution[2].item(),
                    'action_dist_right': action_distribution[1].item(),
                    'action_dist_forward': action_distribution[0].item()
                }
            
            self.training_step += 1
            logging.info({"event": "training_step_completed", "step": self.training_step})
            logging.info({"event": "training_metrics", "policy_loss": f"{policy_loss.item():.4f}", "entropy": f"{entropy.item():.4f}"})
            logging.info({"event": "training_metrics", "avg_return": f"{avg_return.item():.3f}", "max_action_prob": f"{max_action_prob.item():.3f}"})
            logging.info({"event": "action_distribution", "left": f"{action_distribution[2]:.2f}", "right": f"{action_distribution[1]:.2f}", "forward": f"{action_distribution[0]:.2f}"})
            
            return metrics
            
        except Exception as e:
            logging.error({"event": "training_step_error", "exception": e})
            raise
    
    async def SendTrainingBatch(self, request: training_pb2.TrainingBatchRequest, context) -> training_pb2.TrainingBatchResponse:
        """Handle training batch request with proper episode processing."""
        try:
            logging.info({"event": "received_training_batch", "experience_count": len(request.experiences)})
            
            # Deserialize model with error handling
            try:
                buffer = io.BytesIO(request.model_data)
                model = torch.jit.load(buffer, map_location='cpu')
                if not isinstance(model, torch.nn.Module):
                    raise ValueError(f"Deserialized object is not a PyTorch model: {type(model)}")
            except Exception as e:
                logging.error({"event": "failed_to_deserialize_model", "exception": f"{e}"})
                return training_pb2.TrainingBatchResponse(
                    success=False,
                    message=f"Model deserialization failed: {str(e)}",
                    training_step=self.training_step
                )
            
            # Setup model and optimizer
            self.setup_model_and_optimizer(model, request.is_cold_start)
            
            # Organize experiences into episodes
            try:
                episodes = self.organize_experiences_by_episodes(request.experiences)
                if not episodes:
                    raise ValueError("No valid episodes found")
            except Exception as e:
                logging.error({"event": "failed_to_organize_episodes", "exception": e})
                return training_pb2.TrainingBatchResponse(
                    success=False,
                    message=f"Episode organization failed: {str(e)}",
                    training_step=self.training_step
                )
            
            # Process episodes for training
            try:
                states, actions, returns, input_size = self.process_episodes_for_training(episodes)
            except Exception as e:
                logging.error({"event": "failed_to_process_episodes", "exception": e})
                return training_pb2.TrainingBatchResponse(
                    success=False,
                    message=f"Episode processing failed: {str(e)}",
                    training_step=self.training_step
                )
            
            # Train the model
            try:
                metrics = self.perform_training_step(states, actions, returns, input_size)
            except Exception as e:
                logging.error({"event": "training_step_failed", "exception": e})
                return training_pb2.TrainingBatchResponse(
                    success=False,
                    message=f"Training failed: {str(e)}",
                    training_step=self.training_step
                )
            
            # Create model update
            try:
                # Serialize model with proper error handling
                scripted_model = torch.jit.script(self.model)  # or torch.jit.trace()
                buffer = io.BytesIO()
                torch.jit.save(scripted_model, buffer)
                model_data = buffer.getvalue()

                model_update = training_pb2.ModelUpdateResponse(
                    snake_id=request.snake_id,
                    timestamp=datetime.now().isoformat(),
                    training_step=self.training_step,
                    model_data=model_data,
                    metrics=training_pb2.TrainingMetrics(
                        loss=metrics['loss'],
                        entropy_mean=metrics['entropy_mean'],
                        avg_reward=metrics['avg_return'],
                        max_action_prob=metrics['max_action_prob'],
                        num_samples=metrics['num_samples'],
                        positive_rewards=metrics['positive_returns'],
                        negative_rewards=metrics['negative_returns']
                    )
                )
                
                # Send update to all queues for this snake_id
                if request.snake_id in self.model_update_queues:
                    for queue in self.model_update_queues[request.snake_id]:
                        try:
                            await asyncio.wait_for(queue.put(model_update), timeout=1.0)
                            logging.info({"event": "queued_model_update", "snake_id": request.snake_id, "training_step": self.training_step})
                        except asyncio.TimeoutError:
                            logging.warning({"event": "timeout_queuing_model_update", "snake_id": request.snake_id})
                        except Exception as e:
                            logging.warning({"event": "error_queuing_model_update", "exception": f"{e}"})
                
                logging.info({"event": "training_completed", "episode_count": len(episodes)})
                
                return training_pb2.TrainingBatchResponse(
                    success=True,
                    message=f"Training completed on {len(episodes)} episodes",
                    training_step=self.training_step
                )
                
            except Exception as e:
                logging.error({"event": "failed_to_create_model_update", "exception": e})
                return training_pb2.TrainingBatchResponse(
                    success=False,
                    message=f"Model update creation failed: {str(e)}",
                    training_step=self.training_step
                )
            
        except Exception as e:
            logging.error({"event": "error_processing_training_batch", "exception": e})
            logging.error({"event": "traceback", "traceback": traceback.format_exc()})
            return training_pb2.TrainingBatchResponse(
                success=False,
                message=f"Training failed: {str(e)}",
                training_step=self.training_step
            )
    
    async def GetModelUpdates(self, request: training_pb2.ModelUpdateRequest, context):
        """Stream model updates to inference."""
        logging.info({"event": "starting_model_update_stream", "snake_id": request.snake_id})
        
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
                    logging.info({"event": "streaming_model_update", "training_step": model_update.training_step})
                    yield model_update
                except Exception as e:
                    logging.error({"event": "error_getting_model_update_from_queue", "exception": e})
                    break
                
        except asyncio.CancelledError:
            logging.info({"event": "model_update_stream_cancelled", "snake_id": request.snake_id})
        except Exception as e:
            logging.error({"event": "error_in_model_update_stream", "exception": e})
        finally:
            # Clean up: remove this queue from the snake's queue list
            if request.snake_id in self.model_update_queues:
                try:
                    self.model_update_queues[request.snake_id].remove(update_queue)
                    if not self.model_update_queues[request.snake_id]:
                        del self.model_update_queues[request.snake_id]
                except ValueError:
                    pass  # Queue already removed
            logging.info({"event": "model_update_stream_ended", "snake_id": request.snake_id})

async def serve(port: int, snake_id: str, learning_rate: float):
    """Start the gRPC server."""
    # Create the training service
    training_service = GRPCTrainingService(snake_id, learning_rate)
    
    # Create server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add the service
    training_pb2_grpc.add_TrainingServiceServicer_to_server(training_service, server)
    
    logging.info({"event": "starting_grpc_server", "port": port})
    
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    
    await server.start()
    logging.info({"event": "grpc_server_started", "listen_addr": listen_addr})
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info({"event": "shutting_down_grpc_server"})
        await server.stop(5)

async def main():
    parser = argparse.ArgumentParser(description="gRPC Training Server")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--snake_id", required=True, help="Snake ID")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--log_file", default="grpc_training.log", help="Log file path")
    
    args = parser.parse_args()
    
    logger.setup_as_default()
    
    await serve(args.port, args.snake_id, args.learning_rate)

if __name__ == "__main__":
    asyncio.run(main())
