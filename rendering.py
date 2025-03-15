import pygame
from config import *

class GameRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.score_font = pygame.font.SysFont("Arial", 24)
    
    def draw_game_area(self, snake, food):
        # Draw game background
        game_rect = pygame.Rect(0, 0, GAME_WIDTH, GAME_HEIGHT)
        pygame.draw.rect(self.screen, BLACK, game_rect)
        
        # Draw food
        food_rect = pygame.Rect(food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, RED, food_rect)
        
        # Draw snake
        for i, cell in enumerate(snake):
            x, y = cell
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = GREEN if i == 0 else DARKGREEN
            pygame.draw.rect(self.screen, color, rect)
        
        # Draw grid lines
        for x in range(0, GAME_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, GAME_HEIGHT))
        for y in range(0, GAME_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (GAME_WIDTH, y))
    
    def draw_vision_area(self, visible_cells):
        vision_x_offset = GAME_WIDTH + 100
        vision_y_offset = (WINDOW_HEIGHT - (VISION_DISPLAY_ROWS * VISION_CELL_SIZE)) // 2
        
        # Draw vision background
        vision_rect = pygame.Rect(
            vision_x_offset, 
            vision_y_offset, 
            VISION_DISPLAY_COLS * VISION_CELL_SIZE, 
            VISION_DISPLAY_ROWS * VISION_CELL_SIZE
        )
        pygame.draw.rect(self.screen, DARKGRAY, vision_rect)
        
        # Draw visible cells
        for col in range(VISION_DISPLAY_COLS):
            for row in range(VISION_DISPLAY_ROWS):
                cell_rect = pygame.Rect(
                    vision_x_offset + col * VISION_CELL_SIZE,
                    vision_y_offset + row * VISION_CELL_SIZE,
                    VISION_CELL_SIZE, 
                    VISION_CELL_SIZE
                )
                
                if (col, row) in visible_cells:
                    pygame.draw.rect(self.screen, visible_cells[(col, row)], cell_rect)
                    
                pygame.draw.rect(self.screen, GRAY, cell_rect, 1)
    
    def draw_score(self, episode, avg_score, score):
        score_text = self.score_font.render(
            f"episode: {episode} avg: {avg_score:.2f} score: {score}", 
            True, 
            WHITE
        )
        score_rect = score_text.get_rect(topright=(WINDOW_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
