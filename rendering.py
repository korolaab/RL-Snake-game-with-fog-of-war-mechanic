import pygame
from config import *

class GameRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.score_font = pygame.font.SysFont("Arial", 24)

    def generate_color_by_number(num):
        """
        Color generator based on number, compatible with Pygame
        0 - light blue
        1 - pink
        Other numbers - pseudorandom colors
        
        Args:
            num (int): Number for color generation
        
        Returns:
            tuple: Color in (R, G, B) format for Pygame
        """
        # Predefined colors
        if num == 0:
            return (0, 191, 255)  # Light blue (#00BFFF)
        elif num == 1:
            return (255, 105, 180)  # Pink (#FF69B4)
        else:
            # For other numbers, generate pseudorandom color based on the number
            # Use the number as seed for generation
            random.seed(num)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            return (r, g, b)
    
    def draw_game_area(self, game_field):
        # Draw game background
        game_rect = pygame.Rect(0, 0, GAME_WIDTH, GAME_HEIGHT)
        pygame.draw.rect(self.screen, BLACK, game_rect)
        
        for y, row in enumerate(game_field):          # y — строка
            for x, cell in enumerate(row):     # x — колонка
                if cell['type'] == "food":
                    # Draw food
                    food_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, RED, food_rect)
        
                # Draw snakes
                if cell['type'] == "snake":
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    color = DARKGREEN #if i == 0 else DARKGREEN
                    pygame.draw.rect(self.screen, color, rect)

                if cell['type'] == "snake_head":
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    color = GREEN
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
