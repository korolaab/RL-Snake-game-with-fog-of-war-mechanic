import pygame
import sys
from config import *
from game import SnakeGame
from rendering import GameRenderer
from agent import PolicyAgent
import torch
import csv
import argparse
from collections import deque
import numpy as np
import cProfile
import mlflow
import json

def main():
    #mlflow init
    mlflow.set_tracking_uri(uri=args.mlflow_server)
    mlflow.set_experiment(args.mlflow_experiment_name)

    pygame.init()
    
    # Initialize screen only if rendering is enabled
    screen = None
    if not args.no_render:
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake + Vision (Policy Gradient with Batch Update)")
    
    clock = pygame.time.Clock()
    
    # Initialize game
    game = SnakeGame()
    
    # Initialize renderer only if rendering is enabled
    renderer = None
    if not args.no_render:
        renderer = GameRenderer(screen)
    
    # Initialize agent
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net_params={"hidden_units_1": args.hidden_units_1,
                "activation_1": args.activation_1,
                "hidden_units_2": args.hidden_units_2,
                "activation_2": args.activation_2,
                "dropout_rate": args.dropout_rate
                }

    #TODO
    
    agents = [ PolicyAgent(
        input_shape=(62, 2),
        num_actions=3,
        device=device,
        agent_id=i,
        lr=args.learning_rate,
        gamma=args.gamma,
        beta=args.beta,
        update_interval=args.update_interval,
        params=net_params
    ) for i in range(args.N_snakes)]
    
    # Setup score tracking
    with open("score_history.csv", "w") as f:
        f.write("episode,score\n")
    
    episode = 0
    avg_score = 0
    score_queue = deque(maxlen=100)
    ticks = 0
    score = 0
    last_action = 0
    steps_without_improvement = 0
    max_snake_len = 0
    max_avg_score = 0
    max_score = 0
    done = False
    early_stoping = False
    episodes_without_score_improvement = 0
    actions_map = {0: "straight", 1: "left", 2: "right"}
    fps = FPS
    with mlflow.start_run(): 
        mlflow.log_param('net_params', json.dumps(net_params))
        while episode < args.episodes and early_stoping == False:
            if not args.no_render:
                screen.fill(BLACK)
            
            # Agent selects action
            for agent in agents:
                # Get state information
                visible_cells = game.get_visible_cells(agent.id)
                state_matrix = game.get_state_matrix(visible_cells, last_action, agent.id)
                action = agent.select_action(state_matrix)
                last_action = action
                move = actions_map[action]
            
                # Handle events only if rendering is enabled
                if not args.no_render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_LEFT and args.N_snakes == 1:
                                move = "left"
                            elif event.key == pygame.K_RIGHT and args.N_snakes == 1:
                                move = "right"
                            if event.key == pygame.K_UP:
                                fps = 10 if fps > 10 else FPS
                                
                # Update game state
                reward, game_over = game.update(move, ticks, agent.id)
                done = max(done, game_over)
                agent.store_reward(reward)
                
            score += game.food_position_update()
            ticks += 1
            # Check if score is stuck
            if max_score < score:
                max_score = score
                steps_without_improvement = 0
            else:
                steps_without_improvement+= 1

            if steps_without_improvement > args.steps_without_improvement:
                done = True
                print(f"Terminating episode: No score improvement for {args.steps_without_improvement} steps")
            
            # Handle episode completion
            if done:
                for agent in agents:
                    agent.finish_episode()

                with open("score_history.csv", "a") as f:
                    f.write(f"{episode},{score}\n")

                mlflow.log_metric("score", score,step=episode)
                
                episode += 1
                score_queue.append(score)
                np_score_queue = np.array(score_queue)
                low_p = np.percentile(np_score_queue, 10)
                high_p = np.percentile(np_score_queue, 90)
                std = np_score_queue.std()
                if std > 1 and len(np_score_queue) > 5:
                    np_score_queue = np_score_queue[(np_score_queue > low_p) & (np_score_queue < high_p)]
                avg_score = np_score_queue.mean()
                mlflow.log_metric("avg_score", avg_score,step=episode)
                #print(f"{episode} avg_score={avg_score}")
                
                # Check if avg_score is not growing
                if max_avg_score < avg_score:
                    max_avg_score = avg_score
                    episodes_without_score_improvement = 0
                else:
                    episodes_without_score_improvement+= 1

                if episodes_without_score_improvement > args.episodes_without_score_improvement:
                    early_stoping = True
                    print(f"Terminating game: No avg_score improvement for {args.episodes_without_score_improvement} steps")
                game.reset()
                done = False
                score = 0
                ticks = 0
                steps_without_improvement = 0
                max_snake_len = 0
            
            # Rendering only if rendering is enabled
            if not args.no_render:
                visible_cells = game.get_visible_cells(0)
                renderer.draw_game_area(game.field)
                renderer.draw_vision_area(visible_cells)
                pygame.draw.line(screen, GRAY, (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
                
                #score = max(score, len(game.snake) - 3)
                print(avg_score)
                renderer.draw_score(episode, avg_score, score)
                
                pygame.display.flip()
            else:
                # Update score calculation even when not rendering
                score = max(score, len(game.snake) - 3)
            
            # Use a very high FPS if no_render is true for faster execution
            if args.no_render:
                clock.tick(0)  # Run as fast as possible
            else:
                clock.tick(1)
            
    return max_avg_score, episode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an RL agent to play snake')
    
    # Add hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='Discount factor')
    parser.add_argument('--beta', type=float, default=0.1, help='Exploration rate')
    parser.add_argument('--update_interval', type=int, default=1, help='Policy update interval')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--hidden_units_1', type=int, default=8, help='Hidden units in first layer')
    parser.add_argument('--hidden_units_2', type=int, default=16, help='Hidden units in second layer')
    parser.add_argument('--activation_1', type=str, default='Tanh', 
                        help='Activation function for first hidden layer')
    parser.add_argument('--activation_2', type=str, default='Tanh', 
                        help='Activation function for second hidden layer')
    
    # Add other parameters
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--run_id', type=str, default=None, help='Unique ID for this run')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--params_file', type=str, default=None, 
                        help='JSON file containing hyperparameters (overrides command line args)')
    parser.add_argument('--steps_without_improvement', type=int, default=10000, 
                        help='Maximum number of steps allowed without score improved')
    parser.add_argument('--episodes_without_score_improvement', type=int, default=100, 
                        help='Maximum number of episodes allowed without avg_score improved')
    parser.add_argument('--no_render', action='store_true', help='No rendering mode')
    parser.add_argument('--cProfile', action='store_true', help='Run cProfile on main function')
    parser.add_argument('--mlflow_server', type=str, default=None, help='Mlflow server host')
    parser.add_argument('--mlflow_experiment_name', required=True, type=str, help='Mlflow experiment')
    parser.add_argument('--N_snakes', type=int, default=1, help='Number of snakes playing the Game')

    args = parser.parse_args()
    if args.cProfile == True:
        cProfile.run('main()', 'main_rpofile_output.prof')
    else:
        avg_score, episodes = main()
        print(f"Average Score: {avg_score}")
        print(f"Completed Episodes: {episodes}")
