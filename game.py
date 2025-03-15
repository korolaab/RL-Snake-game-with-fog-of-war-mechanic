import random
import numpy as np
from config import *

class SnakeGame:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.snake = [
            (GRID_WIDTH // 2, GRID_HEIGHT // 2),
            (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
            (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
        ]
        self.direction = (1, 0)
        self.food = self.random_food_position()
    
    def random_food_position(self):
        while True:
            pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if pos not in self.snake:
                return pos
    
    def relative_turn(self, turn_command):
        if turn_command == "left":
            return (self.direction[1], -self.direction[0])
        elif turn_command == "right":
            return (-self.direction[1], self.direction[0])
        else:
            return self.direction
    
    def update(self, move, ticks):
        # Apply the move
        self.direction = self.relative_turn(move)
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)
        
        # Check for collision with self
        if new_head in self.snake or len(self.snake) == 1:
            return 0, True  # Game over
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check if snake ate food
        reward = 1
        if new_head == self.food:
            reward += 1
            self.food = self.random_food_position()
        else:
            # Remove tail unless periodic tick condition or food was eaten
            if ticks == 50:
                self.snake.pop()
                ticks = 0
            self.snake.pop()
        
        return reward, False
    
    def get_visible_cells(self):
        head_x, head_y = self.snake[0]
        visible_cells = {}
        
        # Rotation matrix based on current direction
        if self.direction == (0, -1):      # Up
            def rotate(dx, dy): return (dx, dy)
        elif self.direction == (1, 0):     # Right
            def rotate(dx, dy): return (dy, -dx)
        elif self.direction == (0, 1):     # Down
            def rotate(dx, dy): return (-dx, -dy)
        elif self.direction == (-1, 0):    # Left
            def rotate(dx, dy): return (-dy, dx)
        else:
            def rotate(dx, dy): return (dx, dy)
        
        # Calculate visible cells
        for dx in range(-VISION_RADIUS, VISION_RADIUS + 1):
            for dy in range(-VISION_RADIUS, VISION_RADIUS + 1):
                if abs(dx) + abs(dy) > VISION_RADIUS:
                    continue
                    
                r_x, r_y = rotate(dx, dy)
                disp_col = (VISION_DISPLAY_COLS // 2) + r_x
                disp_row = (VISION_DISPLAY_ROWS // 2) + r_y
                
                if not (0 <= disp_col < VISION_DISPLAY_COLS and 0 <= disp_row < VISION_DISPLAY_ROWS):
                    continue
                    
                global_x = (head_x + dx) % GRID_WIDTH
                global_y = (head_y + dy) % GRID_HEIGHT
                cell = (global_x, global_y)
                
                if (dx, dy) == (0, 0):
                    color = GREEN
                elif cell in self.snake:
                    color = DARKGREEN
                elif cell == self.food:
                    color = RED
                else:
                    color = WHITE
                    
                visible_cells[(disp_col, disp_row)] = color
                
        return visible_cells
    
    def get_state_matrix(self, visible_cells, last_action):
        # Convert visible cells to numerical matrix for neural network input
        matrix = []
        for (col, row), color in visible_cells.items():
            if color == DARKGREEN:
                matrix.append([1, 0])
            elif color == RED:
                matrix.append([0, 1])
            elif color == WHITE:
                matrix.append([0, 0])
        
        # Add snake length information
        is_alive = np.exp(-np.abs(len(self.snake)))
        matrix.append([is_alive, 1 - is_alive])
        
        # Add last action information
        last_action_vector = [1, 0] if last_action != 1 else [0, 1]
        matrix.append(last_action_vector)
        
        return np.array(matrix)
