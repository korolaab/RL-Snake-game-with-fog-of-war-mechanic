import requests
import time
import json
import argparse
import sys
import os
from snake_agent import SnakeAgent
from datetime import datetime
import logging
import logger


def get_new_state(base_url, last_seen_frame):
    """Get state - synchronous HTTP request (minimal duplicate checking)"""
    try:
        response = requests.get(base_url, timeout=5)
        response.raise_for_status()
        state = response.json()
        
        episode = state.get('episode')
        frame = state.get('frame')
        current_frame = (episode, frame)
        
        # In synchronous mode, always process the state since frames advance only on moves
        # Duplicate detection is less critical since agent controls timing
        logging.debug({"event": "received_state", "episode": episode, "frame": frame})
        return state, current_frame
        
    except requests.RequestException as e:
        logging.error({"event": "failed_to_get_state", "error": str(e)})
        return None, last_seen_frame
def send_move(move_url, move: str):
    """Send control action."""
    payload = {"move": move}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(move_url, json=payload, headers=headers)
        response.raise_for_status()
        logging.info({"event": "sent_move", "move": move})
    except requests.RequestException as e:
        logging.error({"event": "error_sending_move", "exception": str(e)})

def neural_agent_local(snake_id: str, log_file: str, env_host: str,
                           model_save_dir: str = "models", learning_rate: float = 0.001,
                           batch_size: int = 5, gamma: float = 0.99, beta: float = 0.1,
                           max_episodes: int = None):
    """
    Neural agent with LOCAL REINFORCE training only.
    batch_size = number of episodes before training batch
    """
    base_url = f"http://{env_host}/snake/{snake_id}"
    move_url = f"{base_url}/move"
    reset_url = f"http://{env_host}/reset"

    logging.info({"event": "starting_neural_agent_local", "snake_id": snake_id})
    logging.info({"event": "batch_size_configured", "batch_size": batch_size, "unit": "episodes"})
    logging.info({"event": "mode_configured", "mode": "collect_episodes_then_batch_train"})

    # Read N_EPISODES from param, env var, or default
    if max_episodes is None:
        env_n_episodes = os.environ.get('N_EPISODES')
        if env_n_episodes:
            max_episodes = int(env_n_episodes)
        else:
            max_episodes = 10000000

    # Create agent
    agent = SnakeAgent(
            snake_id=snake_id,
            model_save_dir=model_save_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            beta=beta
    )

    # Output model info
    model_info = agent.get_model_info()
    logging.info({"event": "agent_initialized", "model_info": model_info})
    try:
        episode_counter = 0
        last_seen_frame = None
        
        # Legacy early stopping based on snake length improvement
        steps_without_improvement = 0
        max_snake_len = 0
        STEPS_WITHOUT_IMPROVEMENT_LIMIT = 10000
        
        logging.info({"event": "starting_synchronous_agent", "snake_id": snake_id})
        logging.info({"event": "early_stopping_config", "steps_limit": STEPS_WITHOUT_IMPROVEMENT_LIMIT})
        
        while True:
            logging.info({"event": "starting_episode", "episode": episode_counter})
            previous_action = "forward"
            
            # Reset early stopping variables for new episode
            steps_without_improvement = 0
            max_snake_len = 0
            
            try:
                while True:
                    # 1. Get current state (synchronous)
                    state, last_seen_frame = get_new_state(base_url, last_seen_frame)
                    
                    if state is None:
                        # No new state, brief wait before retry
                        time.sleep(0.01)
                        continue
                    
                    # 2. Extract and validate state data
                    episode_count = state.get("episode", 'null')
                    frame_count = state.get("frame", 'null')
                    visible_cells = state.get("visible_cells", 'null')
                    reward = state.get("reward", 'null')
                    game_over = state.get("game_over", 'null')
                    
                    if any(x == 'null' for x in [visible_cells, episode_count, frame_count, reward, game_over]):
                        logging.error({"event": "invalid_state_received", "state": state})
                        continue
                    
                    logging.info({"event": "state_received",
                                "visible_cells": visible_cells,
                                "episode": episode_count,
                                "frame": frame_count,
                                "reward": reward,
                                "game_over": game_over})
                    
                    # Legacy early stopping: track snake length improvement
                    current_snake_length = state.get("snake_length", 3)
                    if max_snake_len < current_snake_length:
                        max_snake_len = current_snake_length
                        steps_without_improvement = 0
                        logging.debug({"event": "snake_length_improved", "new_max": max_snake_len})
                    else:
                        steps_without_improvement += 1
                    
                    if steps_without_improvement > STEPS_WITHOUT_IMPROVEMENT_LIMIT:
                        logging.info({"event": "early_stopping_triggered", 
                                    "steps_without_improvement": steps_without_improvement,
                                    "max_snake_length": max_snake_len})
                        game_over = True  # Force episode end
                    
                    # 3. Add experience to agent
                    should_send_batch = agent.add_experience(
                        state=state,
                        action=previous_action,
                        reward=reward,
                        done=game_over
                    )
                    
                    # 4. Check game over
                    
                    if game_over == True:
                        episode_counter += 1
                        logging.info({"event": "episode_ended", "episode": episode_count, "total_agent_episodes": episode_counter})
                        
                        if max_episodes is not None and episode_counter >= max_episodes:
                            logging.info({"event": "inference_max_episodes_completed", "max_episodes": max_episodes})
                            sys.exit(0)
                            
                        if should_send_batch:
                            logging.info({"event": "sending_batch", "episodes_completed": agent.batch_size})
                            success = agent.send_training_batch_and_wait()
                            if success:
                                logging.info({"event": "received_improved_model", "source": "batch_training"})
                            else:
                                logging.error({"event": "batch_training_failed"})
                        else:
                            completed_episodes = len(agent.completed_episodes)
                            remaining = agent.batch_size - completed_episodes
                            logging.info(f"📊 Episode added to batch ({completed_episodes}/{agent.batch_size}). "
                                       f"Need {remaining} more episodes before training.")
                        
                        # Reset environment
                        try:
                            reset_response = requests.post(reset_url, timeout=5)
                            if reset_response.status_code == 200:
                                logging.info({"event": "environment_reset_successful"})
                            else:
                                logging.warning({"event": "reset_failed", "status_code": reset_response.status_code})
                        except Exception as reset_error:
                            logging.error({"event": "error_resetting_environment", "exception": str(reset_error)})
                        
                        last_seen_frame = None  # Reset frame tracking
                        break
                    
                    # 5. Predict action
                    action = agent.predict_action(state)
                    
                    # 6. Send move (this advances the game) - ALL actions in synchronous mode
                    try:
                        move_response = requests.post(move_url, json={"move": action}, timeout=5)
                        move_response.raise_for_status()
                        logging.info({"event": "sent_move", "move": action})
                    except requests.RequestException as e:
                        logging.error({"event": "error_sending_move", "exception": str(e)})
                    
                    previous_action = action
                    
            except Exception as game_error:
                logging.error({"event": "error_during_episode", "episode_count": episode_counter, "exception": str(game_error)})
                try:
                    if len(agent.completed_episodes) > 0:
                        logging.info({"event": "sending_partial_batch_due_to_error", "episode_count": len(agent.completed_episodes)})
                        agent.send_training_batch_and_wait()
                    saved_files = agent.save_all_data()
                    logging.info({"event": "data_saved_after_episode_error", "saved_files": saved_files})
                except Exception as save_error:
                    logging.error({"event": "error_saving_data_after_episode_error", "exception": str(save_error)})
                    
    except KeyboardInterrupt:
        logging.info({"event": "keyboard_interrupt", "action": "shutting_down"})
    finally:
        logging.info({"event": "agent_shutdown_complete"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural network agent local training only (no gRPC)")
    parser.add_argument("--snake-id", type=str, required=True, help="Snake ID for this agent")
    parser.add_argument("--log-file", type=str, default="agent_log.json", help="Log file path")
    parser.add_argument("--env-host", type=str, default="localhost:5000", help="Environment host URL")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=5, help="Episodes per batch")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (gamma) for RL")
    parser.add_argument("--beta", type=float, default=0.1, help="Entropy bonus (beta)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Number of episodes before exit (overrides env N_EPISODES)")

    args = parser.parse_args()

    # Configure logging
    logging.getLogger("urllib3").propagate = False
    logger.setup_as_default(container="inference")

    
    neural_agent_local(
            snake_id=args.snake_id,
            log_file=args.log_file,
            env_host=args.env_host,
            model_save_dir=args.model_dir,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            beta=args.beta,
            max_episodes=args.max_episodes
        )

