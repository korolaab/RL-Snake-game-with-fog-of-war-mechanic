import random
import numpy as np
from config import *

class Snake(list):
    def __init__(self, snake_id, direction = (1,0), start_length = 4):
        start_tail = [
            (GRID_WIDTH // 2 - i, GRID_HEIGHT // 2) for i in range(start_length)
        ]
        super().__init__(start_tail)
        
        self.direction = direction
        self.id = snake_id
        self.reward = 0

    def step(self):
        # Calculate new head position
        head_x, head_y = self[0]
        dx, dy = self.direction
        new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)

        self.insert(0, new_head)

    @property
    def head(self):
        return self[0]

    @property
    def tail(self):
        return self[1:]

class SnakeGame:
    def __init__(self, N_snakes=1):
        self.N_snakes = N_snakes
        self.field = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.reset()
        self.food_is_eaten = False
    
    def reset(self):
        self.snakes = [Snake(i) for i in range(self.N_snakes)]
    
        self.direction = (1, 0)
        self.food = self.random_food_position()
        self.fill_game_grid()

    def fill_game_grid(self):
        self.field = [[{"type": None, "id": 0} for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        for snake in self.snakes:
            self.field[snake.head[0]][snake.head[1]] = {"type": "snake_head", "id": snake.id} 
            for segment in snake.tail:
                #if self.field[ segment[0] ][segment[1]]["type"] == "snake":
                #    return True
                self.field[segment[0]][segment[1]] = {"type": "snake", "id": snake.id} 
        self.field[self.food[0]][self.food[1]] = {"type": "food", "id": 0}
        return False
    
    def random_food_position(self):
        while True:
            pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if pos not in [x for snake in self.snakes for x in snake]:
                return pos
    
    def relative_turn(self, direction, turn_command):
        if turn_command == "left":
            return (direction[1], -direction[0])
        elif turn_command == "right":
            return (-direction[1], direction[0])
        else:
            return direction
    
    def update(self, move, ticks, agent_id):
        snake = self.snakes[agent_id]

        # Apply the move
        snake.direction = self.relative_turn(snake.direction, move)
        snake.step()
        
        # Check if snake head inside snake tail
        if snake.head in snake.tail or len(snake) == 1:
            return 0, True  # Game over

        snake.reward = 1
        # Check if snake ate food
        if snake.head == self.food or self.food_is_eaten:
            snake.reward += 1
            self.food_is_eaten = True
        else:
            # Remove tail unless periodic tick condition or food was eaten
        #    if ticks == 50:
        #        self.snake.pop()
        #        ticks = 0
            snake.pop()

        self.fill_game_grid()
        
        return snake.reward, False


    def food_position_update(self):
        if self.food_is_eaten == True:
            self.food = self.random_food_position()
            self.food_is_eaten = False
            return 1
        else:
            return 0

    
    def get_visible_cells(self, snake_id):
        snake = self.snakes[snake_id]
        head_x, head_y = snake.head
        visible_cells = {}
        
        # Rotation matrix based on current direction
        if snake.direction == (0, -1):      # Up
            def rotate(dx, dy): return (dx, dy)
        elif snake.direction == (1, 0):     # Right
            def rotate(dx, dy): return (dy, -dx)
        elif snake.direction == (0, 1):     # Down
            def rotate(dx, dy): return (-dx, -dy)
        elif snake.direction == (-1, 0):    # Left
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
                 
                cell = self.field[global_x][global_y]

                if (dx, dy) == (0, 0):
                    color = GREEN
                elif cell['type'] == "snake":
                    color = DARKGREEN
                elif cell['type'] == "food":
                    color = RED
                else:
                    color = WHITE
                    
                visible_cells[(VISION_DISPLAY_COLS - disp_col-1, disp_row)] = color
                
        return visible_cells
    
    def get_state_matrix(self, visible_cells, last_action, snake_id):
        snake = self.snakes[snake_id]
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
        is_alive = np.exp(-np.abs(len(snake)))
        matrix.append([is_alive, 1 - is_alive])
        
        # Add last action information
        last_action_vector = [1, 0] if last_action != 1 else [0, 1]
        matrix.append(last_action_vector)
        
        return np.array(matrix)
