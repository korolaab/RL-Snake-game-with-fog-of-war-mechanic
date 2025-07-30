import torch
import logging
import json
from datetime import datetime
from snake_model import ModelManager
from state_processor import StateProcessor
from data_manager import DataManager
import io
import traceback
import sys

class SnakeAgent:
    """Snake Agent with local REINFORCE training only (no gRPC)."""
    def __init__(self, snake_id: str, model_save_dir: str = "models", learning_rate: float = 0.001,
                 batch_size: int = 5, gamma: float = 0.99, beta: float = 0.1):
        self.snake_id = snake_id
        self.model_save_dir = model_save_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size  # Number of EPISODES before sending batch
        self.gamma = gamma
        self.beta = beta
        
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
        
        # Load existing model
        self.load_existing_model()
    
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
            self.model = self.model_manager.create_new_model(
                input_size
            )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.model_initialized = True
            self.is_cold_start = True
            logging.info({"event": "created_fresh_model_for_cold_start", "input_size": input_size})
        else:
            if not self.model_manager.validate_input_size(self.model, input_size):
                actual_size = self.model_manager.get_model_input_size(self.model)
                logging.error({"event": "input_size_mismatch", "expected": actual_size, "got": input_size})
                self.is_cold_start = True
    
    def predict_action(self, state):
        """Predict action based on state.""" 
        state_tensor = self.state_processor.process_state(state)
        flat_tensor = state_tensor.flatten().unsqueeze(0)
        input_size = flat_tensor.shape[1]
        self.ensure_model_initialized(input_size)
        with torch.no_grad():
            self.model.eval()
            action_probs = self.model(flat_tensor)
            m = torch.distributions.Categorical(action_probs)
            action_idx = m.sample()
            self.model.train()
        predicted_action = self.actions[action_idx]
        logging.info({"event": "predicted_action",
                        "action": predicted_action,
                        "probabilities": action_probs.numpy().tolist()})
        return predicted_action
               
    def add_experience(self, state, action, reward, next_state=None, done=False):
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
        if done:
            return self._complete_episode()
        return False

    def _complete_episode(self):
        if not self.current_episode_experiences:
            logging.warning({"event": "no_experiences_in_current_episode"})
            return False
        episode_data = {
            'episode_number': self.current_episode,
            'experiences': self.current_episode_experiences.copy(),
            'total_steps': len(self.current_episode_experiences),
            'total_reward': sum(exp['reward'] for exp in self.current_episode_experiences),
            'final_reward': self.current_episode_experiences[-1]['reward'] if self.current_episode_experiences else 0
        }
        self.completed_episodes.append(episode_data)
        logging.info(f"🎮 Episode {self.current_episode} completed: {len(self.current_episode_experiences)} steps, "
                    f"total_reward={episode_data['total_reward']}, final_reward={episode_data['final_reward']}")
        self.current_episode_experiences = []
        self.current_episode += 1
        self.current_episode_steps = 0
        should_send_batch = len(self.completed_episodes) >= self.batch_size
        if should_send_batch:
            logging.info({"event": "ready_to_send_batch", "completed_episodes": len(self.completed_episodes)})
        return should_send_batch
    
    def send_training_batch_and_wait(self):
        """Train model on accumulated batch of completed episodes."""
        if len(self.completed_episodes) == 0:
            logging.warning({"event": "no_completed_episodes_to_send"})
            return False
        if not self.model_initialized:
            logging.warning({"event": "model_not_initialized", "action": "cannot_train_batch"})
            return False
        try:
            total_episodes = len(self.completed_episodes)
            total_experiences = sum(len(ep['experiences']) for ep in self.completed_episodes)
            logging.info({"event": "local_training_start", "total_episodes": total_episodes, "total_experiences": total_experiences, "gamma": self.gamma, "beta": self.beta, "learning_rate": self.learning_rate})
            episodes = [ep['experiences'] for ep in self.completed_episodes]
            all_states, all_actions, all_returns = [], [], []
            for ep_idx, episode in enumerate(episodes):
                episode_states, episode_actions, episode_rewards = [], [], []
                for exp in episode:
                    state_tensor = self.state_processor.process_state(exp['state']).flatten()
                    action_encoding = self.actions
                    action_idx = action_encoding.index(exp['action'])
                    episode_states.append(state_tensor)
                    episode_actions.append(action_idx)
                    episode_rewards.append(exp['reward'])
                if not episode_states:
                    continue
                episode_returns = []
                discounted_return = 0
                for reward in reversed(episode_rewards):
                    discounted_return = reward + self.gamma * discounted_return
                    episode_returns.insert(0, discounted_return)
                episode_returns = torch.tensor(episode_returns, dtype=torch.float32)
                if len(episode_returns) > 1:
                    episode_returns = (episode_returns - episode_returns.mean()) / (episode_returns.std() + 1e-8)
                all_states.extend(episode_states)
                all_actions.extend(episode_actions)
                all_returns.extend(episode_returns.tolist())
            if not all_states:
                logging.error({"event": "no_states_to_train"})
                return False
            max_length = max(len(s) for s in all_states)
            padded_states = [torch.cat([s, torch.zeros(max_length - len(s))]) if len(s) < max_length else s for s in all_states]
            states_tensor = torch.stack(padded_states)
            actions_tensor = torch.tensor(all_actions, dtype=torch.long)
            returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
            self.model.train()
            action_probs = self.model(states_tensor)
            m = torch.distributions.Categorical(action_probs)
            log_probs = m.log_prob(actions_tensor)
            entropy = m.entropy().mean()
            policy_loss = -(log_probs * returns_tensor).mean()
            total_loss = policy_loss - self.beta * entropy
            if torch.isnan(total_loss):
                logging.critical({"event": "training_loss_nan_abort"})
                return False
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            avg_return = returns_tensor.mean().item()
            max_action_prob = action_probs.max(dim=-1)[0].mean().item()
            action_counts = torch.bincount(actions_tensor, minlength=3)
            action_distribution = action_counts.float() / len(actions_tensor)
            metrics = {
                'loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'entropy_mean': entropy.item(),
                'avg_return': avg_return,
                'max_action_prob': max_action_prob,
                'num_samples': len(states_tensor),
                'positive_returns': (returns_tensor > 0).sum().item(),
                'negative_returns': (returns_tensor <= 0).sum().item(),
                'action_dist_left': action_distribution[2].item(),
                'action_dist_right': action_distribution[1].item(),
                'action_dist_forward': action_distribution[0].item(),
            }
            logging.info({
                'event': 'local_training_step_summary',
                'step': self.batch_number + 1,
                **metrics,
                'action_distribution': {
                    'left': f"{action_distribution[2]:.2f}",
                    'right': f"{action_distribution[1]:.2f}",
                    'forward': f"{action_distribution[0]:.2f}"
                }
            })
            self.completed_episodes = []
            self.batch_number += 1
            if self.is_cold_start:
                self.is_cold_start = False
                logging.info({"event": "cold_start_completed"})
            return True
        except Exception as e:
            logging.error({"event": "local_training_error", "exception": str(e)})
            return False
    
    def save_experience(self, state, reward, action):
        return self.add_experience(state, action, reward)
    def save_all_data(self):
        try:
            saved_files = {}
            if self.current_episode_experiences:
                logging.info({"event": "completing_partial_episode", "episode": self.current_episode, "experience_count": len(self.current_episode_experiences)})
                if self.current_episode_experiences:
                    self.current_episode_experiences[-1]['done'] = True
                self._complete_episode()
            if self.model is not None:
                model_path = self.model_manager.save_model(
                    self.model, self.optimizer, self.snake_id
                )
                saved_files['model'] = model_path
            data_files = self.data_manager.save_all_data(self.model, self.snake_id)
            saved_files.update(data_files)
            stats = self.data_manager.get_statistics()
            logging.info({"event": "game_session_completed", "statistics": stats})
            logging.info({"event": "total_episodes_completed", "count": self.current_episode})
            logging.info({"event": "total_steps_across_all_episodes", "count": self.total_steps})
            return saved_files
        except Exception as e:
            logging.error({"event": "error_saving_data", "exception": srt(e)})
            raise
    def get_model_info(self):
        info = {
            'model_initialized': self.model_initialized,
            'snake_id': self.snake_id,
            'actions': self.actions,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'beta': self.beta,
            'is_cold_start': self.is_cold_start,
            'batch_size': self.batch_size,
            'completed_episodes': len(self.completed_episodes),
            'current_episode': self.current_episode,
            'current_episode_steps': len(self.current_episode_experiences),
            'batch_number': self.batch_number,
            'total_steps': self.total_steps,
        }
        if self.model is not None:
            info['input_size'] = self.model_manager.get_model_input_size(self.model)
        if self.model_info is not None:
            info['loaded_from'] = self.model_info
        return info
    def get_statistics(self):
        stats = self.data_manager.get_statistics()
        stats.update({
            'episodes_completed': self.current_episode,
            'episodes_in_current_batch': len(self.completed_episodes),
            'current_episode_steps': len(self.current_episode_experiences),
            'total_steps': self.total_steps
        })
        return stats
