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


class RunningApple:
    """Apple that runs away from snake at half speed"""
    
    def __init__(self, position, grid_width, grid_height):
        self.position = position
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.move_counter = 0
        
    def should_move(self):
        """Check if apple should move (every 2 frames)"""
        self.move_counter += 1
        if self.move_counter >= 2:
            self.move_counter = 0
            return True
        return False
    
    def get_escape_direction(self, snake_head, snake_body):
        """Calculate best escape direction away from snake"""
        # Calculate vector away from snake head
        dx = self.position[0] - snake_head[0]
        dy = self.position[1] - snake_head[1]
        
        # Handle wrapping distances
        if abs(dx) > self.grid_width / 2:
            dx = -dx / abs(dx) * (self.grid_width - abs(dx))
        if abs(dy) > self.grid_height / 2:
            dy = -dy / abs(dy) * (self.grid_height - abs(dy))
        
        # Normalize direction
        if abs(dx) > abs(dy):
            move_dir = (1 if dx > 0 else -1, 0)
        elif abs(dy) > abs(dx):
            move_dir = (0, 1 if dy > 0 else -1)
        else:
            # Equal distance, prefer horizontal movement
            move_dir = (1 if dx > 0 else -1, 0)
        
        # Check if proposed move is valid (not into snake body)
        new_pos = ((self.position[0] + move_dir[0]) % self.grid_width,
                   (self.position[1] + move_dir[1]) % self.grid_height)
        
        if new_pos not in snake_body:
            return move_dir
        
        # If blocked, try alternative directions
        alternatives = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for alt_dir in alternatives:
            alt_pos = ((self.position[0] + alt_dir[0]) % self.grid_width,
                      (self.position[1] + alt_dir[1]) % self.grid_height)
            if alt_pos not in snake_body:
                return alt_dir
        
        # If all directions blocked, don't move
        return (0, 0)
    
    def move(self, snake_head, snake_body):
        """Move apple away from snake if it's time to move"""
        if self.should_move():
            move_dir = self.get_escape_direction(snake_head, snake_body)
            self.position = ((self.position[0] + move_dir[0]) % self.grid_width,
                           (self.position[1] + move_dir[1]) % self.grid_height)
    
    def respawn(self, snake_body):
        """Respawn apple at random location avoiding snake"""
        while True:
            pos = (random.randint(0, self.grid_width - 1), 
                   random.randint(0, self.grid_height - 1))
            if pos not in snake_body:
                self.position = pos
                self.move_counter = 0
                break

class SnakeGame:
    def __init__(self,
                 GRID_WIDTH, 
                 GRID_HEIGHT,
                 VISION_RADIUS,
                 VISION_DISPLAY_COLS,
                 VISION_DISPLAY_ROWS,
                 max_lifetime=10000,
                 max_hunger_steps=150
                 ):
        self.GRID_WIDTH = GRID_WIDTH
        self.GRID_HEIGHT = GRID_HEIGHT
        self.direction = (1, 0)  # Initially moving right
        self.VISION_RADIUS = VISION_RADIUS
        self.VISION_DISPLAY_COLS = VISION_DISPLAY_COLS
        self.VISION_DISPLAY_ROWS = VISION_DISPLAY_ROWS
        self.last_action = 0  # Initially no turn
        self.ticks = 0
        
        # Lifetime system
        self.max_lifetime = max_lifetime
        self.lifetime_steps = 0
        
        # Hunger and apple tracking system
        self.steps_without_apple = 0
        self.max_hunger_steps = max_hunger_steps
        self.eaten_apples = 0
        
        self.reset()
    
    def reset(self):
        self.snake = [
            (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2),
            (self.GRID_WIDTH // 2 - 1, self.GRID_HEIGHT // 2),
            (self.GRID_WIDTH // 2 - 2, self.GRID_HEIGHT // 2)
        ]
        self.direction = (1, 0)
        self.max_len = 0  # Reset max apples eaten
        
        # Initialize running apple
        apple_pos = self.random_food_position()
        self.apple = RunningApple(apple_pos, self.GRID_WIDTH, self.GRID_HEIGHT)
        
        # Reset lifetime and hunger
        self.lifetime_steps = 0
        self.steps_without_apple = 0
        self.eaten_apples = 0
    
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
        # Increment lifetime counter and hunger
        self.lifetime_steps += 1
        self.steps_without_apple += 1
        
        # Check for lifetime expiration
        if self.lifetime_steps >= self.max_lifetime:
            state = self.get_state()
            return state, 0, True  # Die from old age
        
        # Check for hunger death
        if self.steps_without_apple >= self.max_hunger_steps:
            state = self.get_state()
            return state, 0, True  # Die from hunger
        
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
        
        # Move apple away from snake (half speed)
        self.apple.move(self.snake[0], self.snake)
        
        # Check if snake caught apple
        reward = 1
        if new_head == self.apple.position:
            reward += 1  # Give reward but NO growth
            self.eaten_apples += 1  # Track eaten apples
            self.steps_without_apple = 0  # Reset hunger timer
            self.apple.respawn(self.snake)  # Respawn apple at new location
        
        # Always remove tail (no growth, fixed length snake)
        self.snake.pop()
        
        # Update max_len to track maximum apples eaten (for compatibility)
        self.max_len = max(self.eaten_apples, self.max_len)
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
                elif cell == self.apple.position:
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
        
        # Add eaten apples information instead of snake length
        apples_normalized = min(self.eaten_apples / 10.0, 1.0)  # Normalize to 0-1 range
        matrix.append([apples_normalized, 1 - apples_normalized])
        
        # Add last action information
        last_action_vector = [1, 0] if last_action != 1 else [0, 1]
        matrix.append(last_action_vector)
        
        return np.array(matrix)
    
    def get_state(self):

        visible_cells = self.get_visible_cells()
        state_matrix = self.get_state_matrix(visible_cells, last_action=self.last_action)
        return state_matrix