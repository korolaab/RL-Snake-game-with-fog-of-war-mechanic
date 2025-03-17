import pygame
import sys
from config import *
from game import SnakeGame
from rendering import GameRenderer
from agent import PolicyAgent
import torch
import csv
import argparse

def main():
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
    agent = PolicyAgent(
        input_shape=(62, 2),
        num_actions=3,
        device=device,
        lr=args.learning_rate,
        gamma=args.gamma,
        beta=args.beta,
        update_interval=args.update_interval,
        params={"hidden_units_1": args.hidden_units_1,
                "activation_1": args.activation_1,
                "hidden_units_2": args.hidden_units_2,
                "activation_2": args.activation_2,
                "dropout_rate": args.dropout_rate
                }
    )
    
    # Setup score tracking
    with open("score_history.csv", "w") as f:
        f.write("episode,score\n")
    
    episode = 0
    avg_score = 0
    ticks = 0
    score = 0
    last_action = 0
    actions_map = {0: "straight", 1: "left", 2: "right"}
    fps = FPS
    
    while True and episode < args.episodes:
        if not args.no_render:
            screen.fill(BLACK)
        
        # Get state information
        visible_cells = game.get_visible_cells()
        state_matrix = game.get_state_matrix(visible_cells, last_action)
        
        # Agent selects action
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
                    if event.key == pygame.K_LEFT:
                        move = "left"
                    elif event.key == pygame.K_RIGHT:
                        move = "right"
                    if event.key == pygame.K_UP:
                        fps = 10 if fps > 10 else FPS
        
        # Update game state
        reward, done = game.update(move, ticks)
        ticks += 1
        agent.store_reward(reward)
        
        # Handle episode completion
        if done:
            agent.finish_episode()
            with open("score_history.csv", "a") as f:
                f.write(f"{episode},{score}\n")
            
            episode += 1
            avg_score = (avg_score * (episode - 1) + (len(game.snake) - 3)) / episode
            #print(f"{episode} avg_score={avg_score}")
            
            game.reset()
            done = False
            score = 0
            ticks = 0
        
        # Rendering only if rendering is enabled
        if not args.no_render:
            visible_cells = game.get_visible_cells()
            renderer.draw_game_area(game.snake, game.food)
            renderer.draw_vision_area(visible_cells)
            pygame.draw.line(screen, GRAY, (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
            
            score = max(score, len(game.snake) - 3)
            renderer.draw_score(episode, avg_score, score)
            
            pygame.display.flip()
        else:
            # Update score calculation even when not rendering
            score = max(score, len(game.snake) - 3)
        
        # Use a very high FPS if no_render is true for faster execution
        if args.no_render:
            clock.tick(0)  # Run as fast as possible
        else:
            clock.tick(fps)
            
    return avg_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an RL agent to play snake')
    
    # Add hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
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
    parser.add_argument('--no_render', action='store_true', help='No rendering mode')
    
    args = parser.parse_args()

    avg_score = main()
    print(f"Average Score: {avg_score}")
