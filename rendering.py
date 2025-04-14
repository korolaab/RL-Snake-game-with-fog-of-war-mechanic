import pygame
from config import *
import colorsys
import random

class GameRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.score_font = pygame.font.SysFont("Arial", 24)


    def generate_color(self,num, head=False):
        """
        Color generator based on number, compatible with Pygame
        0 - light blue
        1 - pink
        Other numbers - pseudorandom colors
        If head=True, returns a darker version of the color
        
        Args:
            num (int): Number for color generation
            head (bool): If True, returns a darker version of the color
        
        Returns:
            tuple: Color in (R, G, B) format for Pygame
        """
        #print(num)
        # Predefined colors
        if num == 0:
            # Light blue (#00BFFF)
            if head:
                return (0, 120, 180)  # Darker blue
            else:
                return (0, 191, 255)
        elif num == 1:
            # Pink (#FF69B4)
            if head:
                return (180, 60, 120)  # Darker pink
            else:
                return (255, 105, 180)
        else:
            # For other numbers, generate pseudorandom color based on the number
            # Use the number as seed for generation and HSV color system
            
            random.seed(num)
            
            # Generate random hue (0-1)
            hue = random.random()
            
            # Set saturation and value (brightness)
            saturation = 0.7 + random.random() * 0.3  # 0.7-1.0 for vibrant colors
            
            # If head is True, make the color darker by reducing value
            if head:
                value = 0.4 + random.random() * 0.3  # 0.4-0.7 for darker colors
            else:
                value = 0.7 + random.random() * 0.3  # 0.7-1.0 for brighter colors
            
            # Convert HSV to RGB (0-1 range)
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            
            # Convert to 0-255 range for Pygame
            return (int(r * 255), int(g * 255), int(b * 255))  


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
                    color = self.generate_color(cell['id']) #if i == 0 else DARKGREEN
                    pygame.draw.rect(self.screen, color, rect)

                if cell['type'] == "snake_head":
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    color = self.generate_color(cell['id'],head=True) #if i == 0 else DARKGREEN
                    pygame.draw.rect(self.screen, color, rect)
        
        # Draw grid lines
        for x in range(0, GAME_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, GAME_HEIGHT))
        for y in range(0, GAME_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (GAME_WIDTH, y))
    
    def draw_vision_area(self, snakes_viz):
        for i, viz in enumerate(snakes_viz):
            vision_x_offset = GAME_WIDTH + 100 + 100 * i + 200*i
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
                    
                    if (col, row) in viz:
                        if(viz[(col,row)]['type'] == "snake"):
                            color = self.generate_color(viz[(col,row)]['id'])
                        elif(viz[(col,row)]['type'] == "snake_head"):
                            color = self.generate_color(viz[(col,row)]['id'],head=True)
                        elif(viz[(col,row)]['type'] == "food"):
                            color = RED
                        else:
                            color = WHITE
                        
                        pygame.draw.rect(self.screen, color, cell_rect)
                        
                    pygame.draw.rect(self.screen, GRAY, cell_rect, 1)
    
    def draw_score(self, episode, avg_score, score):
        score_text = self.score_font.render(
            f"episode: {episode} avg: {avg_score:.2f} score: {score}", 
            True, 
            WHITE
        )
        score_rect = score_text.get_rect(topright=(WINDOW_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
