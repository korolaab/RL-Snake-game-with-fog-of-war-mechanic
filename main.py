import pygame
import sys
from config import *
from game import SnakeGame
from rendering import GameRenderer
from agent import PolicyAgent
import torch
import csv

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake + Vision (Policy Gradient with Batch Update)")
    clock = pygame.time.Clock()
    
    # Initialize game
    game = SnakeGame()
    renderer = GameRenderer(screen)
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PolicyAgent(
        input_shape=(62, 2),
        num_actions=3,
        device=device,
        lr=0.005,
        gamma=0.5,
        update_interval=1
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
    
    while True:
        screen.fill(BLACK)
        
        # Get state information
        visible_cells = game.get_visible_cells()
        state_matrix = game.get_state_matrix(visible_cells, last_action)
        
        # Agent selects action
        action = agent.select_action(state_matrix)
        last_action = action
        move = actions_map[action]
        
        # Handle events
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
            print(f"{episode} avg_score={avg_score}")
            
            game.reset()
            done = False
            score = 0
            ticks = 0
        
        # Rendering
        visible_cells = game.get_visible_cells()
        renderer.draw_game_area(game.snake, game.food)
        renderer.draw_vision_area(visible_cells)
        pygame.draw.line(screen, GRAY, (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
        
        score = max(score, len(game.snake) - 3)
        renderer.draw_score(episode, avg_score, score)
        
        pygame.display.flip()
        clock.tick(fps)

if __name__ == "__main__":
    main()
