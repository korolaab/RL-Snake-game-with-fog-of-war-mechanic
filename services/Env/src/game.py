import random
import numpy as np
#from config import *

# --- Colors (RGB) ---
WHITE     = (255, 255, 255)
BLACK     = (0, 0, 0)
GREEN     = (0, 255, 0)    # Snake head
DARKGREEN = (0, 155, 0)    # Snake body
RED       = (255, 0, 0)
BLUE      = (0, 0, 255)
GRAY      = (100, 100, 100)
DARKGRAY  = (50, 51, 50)

class SnakeGame:
    def __init__(self,
                 GRID_WIDTH, 
                 GRID_HEIGHT,
                 VISION_RADIUS,
                 VISION_DISPLAY_COLS,
                 VISION_DISPLAY_ROWS,
                 ):
        self.GRID_WIDTH = GRID_WIDTH
        self.GRID_HEIGHT = GRID_HEIGHT
        self.direction = (1, 0)  # Initially moving right
        self.VISION_RADIUS = VISION_RADIUS
        self.VISION_DISPLAY_COLS = VISION_DISPLAY_COLS
        self.VISION_DISPLAY_ROWS = VISION_DISPLAY_ROWS
        self.last_action = 0  # Initially no turn
        self.ticks = 0
        
        self.reset()
    
    def reset(self):
        self.snake = [
            (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2),
            (self.GRID_WIDTH // 2 - 1, self.GRID_HEIGHT // 2),
            (self.GRID_WIDTH // 2 - 2, self.GRID_HEIGHT // 2)
        ]
        self.direction = (1, 0)
        self.max_len = len(self.snake)
        self.food = self.random_food_position()
    
    def random_food_position(self):
        while True:
            pos = (random.randint(0, self.GRID_WIDTH - 1), random.randint(0, self.GRID_HEIGHT - 1))
            if pos not in self.snake:
                return pos
    
    def relative_turn(self, turn_command):
        if turn_command == 1:
            return (self.direction[1], -self.direction[0])
        elif turn_command == 2:
            return (-self.direction[1], self.direction[0])
        else:
            return self.direction
    
    def update(self, move):
        # Apply the move
        self.direction = self.relative_turn(move)
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = ((head_x + dx) % self.GRID_WIDTH, (head_y + dy) % self.GRID_HEIGHT)
        
        # Check for collision with self
        if new_head in self.snake or len(self.snake) == 1:
            state = self.get_state()
            return state,0, True  # Game over
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check if snake ate food
        reward = 1
        if new_head == self.food:
            reward += 1
            self.food = self.random_food_position()
        else:
            # Remove tail unless periodic tick condition or food was eaten
            if self.ticks == 50:
                self.snake.pop()
                self.ticks = 0
            self.snake.pop()
        self.max_len = max(len(self.snake),self.max_len)
        state = self.get_state()
        return state, reward, False
    
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
        for dx in range(-self.VISION_RADIUS, self.VISION_RADIUS + 1):
            for dy in range(-self.VISION_RADIUS, self.VISION_RADIUS + 1):
                if abs(dx) + abs(dy) > self.VISION_RADIUS:
                    continue
                    
                r_x, r_y = rotate(dx, dy)
                disp_col = (self.VISION_DISPLAY_COLS // 2) + r_x
                disp_row = (self.VISION_DISPLAY_ROWS // 2) + r_y
                
                if not (0 <= disp_col < self.VISION_DISPLAY_COLS and 0 <= disp_row < self.VISION_DISPLAY_ROWS):
                    continue
                    
                global_x = (head_x + dx) % self.GRID_WIDTH
                global_y = (head_y + dy) % self.GRID_HEIGHT
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
    
    def get_state(self):

        visible_cells = self.get_visible_cells()
        state_matrix = self.get_state_matrix(visible_cells, last_action=self.last_action)
        return state_matrix