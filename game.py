import random
import numpy as np
from config import *

class Snake(list):
    def __init__(self, snake_id, direction = (1,0), start_length = 4):
        start_tail = [
            (GRID_WIDTH // 2 - i, (GRID_HEIGHT // 2) + snake_id*3) for i in range(start_length)
        ]
        super().__init__(start_tail)
        
        self.direction = direction
        self.id = snake_id
        self.reward = 0
        self.fpv = {}

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
        self.reset()
        self.food_is_eaten = False
        
    def reset(self):
        self.snakes = [Snake(i) for i in range(self.N_snakes)]
    
        self.field = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.score = 0
        self.direction = (1, 0)
        self.food = self.random_food_position()
        self.update_game_grid()

        for snake in self.snakes:
            self.get_visible_cells(snake.id)
    

    def update_game_grid(self):
        self.field = [[{"type": None, "id": 0} for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        food_is_eaten = False
        for snake in self.snakes:
            if snake.head == self.food:
                food_is_eaten = True
                print('apple_eaten')
                self.food = self.random_food_position()
                self.score += 1
            if self.field[snake.head[0]][snake.head[1]]["type"] == "snake_head":
                print(f'snake_head {snake.id} iniside other snake_head {self.field[snake.head[0]][snake.head[1]]["id"]}')
                return True
            self.field[snake.head[0]][snake.head[1]] = {"type": "snake_head", "id": snake.id} 

        for snake in self.snakes: 
            if food_is_eaten == False:
                snake.pop()
            for segment in snake.tail:
                if self.field[segment[0]][segment[1]]["type"] == "snake" or self.field[segment[0]][segment[1]]["type"] == "snake_head":
                        print(f'snake_tail {snake.id} iniside other {self.field[snake.head[0]][snake.head[1]]["type"]} {self.field[snake.head[0]][snake.head[1]]["id"]}')
                        return True
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
        snake.reward = 1
        
        return snake.reward


    def food_position_update(self):
        self.food = self.random_food_position()

    
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
                
                snake.fpv[(VISION_DISPLAY_COLS - disp_col-1, disp_row)] = cell
    
    def get_state_matrix(self, last_action, snake_id):
        snake = self.snakes[snake_id]
        # Convert visible cells to numerical matrix for neural network input
        matrix = []
        for (col, row), cell in snake.fpv.items():
            if cell['type'] == "snake" and cell['id'] == snake_id:
                matrix.append([1, 0, 0])
            elif (cell['type'] == "snake" or cell['type'] == "snake_head") and cell['id'] != snake_id:
                matrix.append([0, 1, 0])
            elif cell['type'] == "food":
                matrix.append([0, 0, 1])
            elif cell['type'] == None:
                matrix.append([0, 0, 0])
        
        
        # Add snake length information
        is_alive = np.exp(-np.abs(len(snake)))
        matrix.append([is_alive, 1 - is_alive, 0])
        
        # Add last action information
        last_action_vector = [1, 0, 0] if last_action != 1 else [0, 1, 0]
        matrix.append(last_action_vector)
        
        return np.array(matrix)
