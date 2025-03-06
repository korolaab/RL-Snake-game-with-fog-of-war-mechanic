import pygame
import random
import sys
import csv
from collections import defaultdict
import numpy as np
from tabulate import tabulate

# --- Game settings ---
CELL_SIZE = 20
GRID_WIDTH = 15
GRID_HEIGHT = 15
GAME_WIDTH = GRID_WIDTH * CELL_SIZE   # 300 pixels
GAME_HEIGHT = GRID_HEIGHT * CELL_SIZE   # 300 pixels
FPS = 5

# --- Snake vision settings ---
VISION_RADIUS = 5
VISION_DISPLAY_COLS = 11  
VISION_DISPLAY_ROWS = 11   
VISION_CELL_SIZE = 20     
VISION_WIDTH = VISION_DISPLAY_COLS * VISION_CELL_SIZE   # 220 pixels
VISION_HEIGHT = VISION_DISPLAY_ROWS * VISION_CELL_SIZE    # 220 pixels

# --- Window settings ---
WINDOW_WIDTH = GAME_WIDTH + VISION_WIDTH + 200  # 300 + 220 = 520 pixels
WINDOW_HEIGHT = GAME_HEIGHT               # 300 pixels

# --- Colors (RGB) ---
WHITE     = (255, 255, 255)
BLACK     = (0, 0, 0)
GREEN     = (0, 255, 0)    # Snake head
DARKGREEN = (0, 155, 0)    # Snake body
RED       = (255, 0, 0)
BLUE      = (0, 0, 255)    # Field boundary
GRAY      = (100, 100, 100)
DARKGRAY  = (50, 51, 50)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake + Vision (Q-learning)")
clock = pygame.time.Clock()
score_font = pygame.font.SysFont("Arial", 24)

def random_food_position(snake):
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake:
            return pos

def turn_snake(current_direction, command):
    mapping = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0)
    }
    new_direction = mapping.get(command, current_direction)
    if (current_direction[0] + new_direction[0] == 0 and 
        current_direction[1] + new_direction[1] == 0) and current_direction != (0, 0):
        return current_direction
    return new_direction

def relative_turn(direction, turn_command):
    if turn_command == "left":
        return (direction[1], -direction[0])
    elif turn_command == "right":
        return (-direction[1], direction[0])
    else:
        return direction

def draw_game_area(snake, food):
    game_rect = pygame.Rect(0, 0, GAME_WIDTH, GAME_HEIGHT)
    pygame.draw.rect(screen, BLACK, game_rect)
    
    food_rect = pygame.Rect(food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, RED, food_rect)
    
    for i, cell in enumerate(snake):
        x, y = cell
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        color = GREEN if i == 0 else DARKGREEN
        pygame.draw.rect(screen, color, rect)
    
    for x in range(0, GAME_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, GAME_HEIGHT))
    for y in range(0, GAME_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (GAME_WIDTH, y))

def compute_visible_state(snake, food, direction):
    head_x, head_y = snake[0]
    visible_cells = {}
    state_str = ""
    
    if direction == (0, -1):
        def rotate(dx, dy): 
            return (dx, dy)
    elif direction == (1, 0):
        def rotate(dx, dy): 
            return (dy, -dx)
    elif direction == (0, 1):
        def rotate(dx, dy): 
            return (-dx, -dy)
    elif direction == (-1, 0):
        def rotate(dx, dy): 
            return (-dy, dx)
    else:
        def rotate(dx, dy):
            return (dx, dy)
    has_apple = 0 
    apple_x = 0
    apple_y = 0 

    snake_ahead_dist = 0
    snake_on_right_dist = 0
    snake_on_left_dist = 0
    
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
            elif cell in snake:
                color = DARKGREEN
                if (dy > 0):
                    snake_ahead_dist = 1
                    #snake_ahead_dist = min(snake_ahead_dist, abs(dy)) 
                if (dy == 0 and dx < 0):
                    snake_on_left_dist = 1
                    #snake_on_left_dist = min(snake_on_left_dist, abs(dx)) 
                if (dy == 0 and dx > 0):
                    snake_on_right_dist = 1
            elif cell == food:
                color = RED
                has_apple = 1
                if dy > 0:
                    apple_y = 1
                if dx > 0:
                    apple_x = 1
                apple_dist = np.sqrt(dy**2 + dx**2)
            else:
                color = WHITE
            
            visible_cells[(disp_col, disp_row)] = color

        state_str = f"{has_apple},{apple_x},{apple_y},{snake_ahead_dist},{snake_on_left_dist},{snake_on_right_dist}"
    return state_str, visible_cells

def draw_vision_area(visible_cells):
    vision_x_offset = GAME_WIDTH + 100
    vision_y_offset = (WINDOW_HEIGHT - (VISION_DISPLAY_ROWS * VISION_CELL_SIZE)) // 2
    vision_rect = pygame.Rect(vision_x_offset, vision_y_offset, VISION_DISPLAY_COLS * VISION_CELL_SIZE, VISION_DISPLAY_ROWS * VISION_CELL_SIZE)
    pygame.draw.rect(screen, DARKGRAY, vision_rect)
    
    for col in range(VISION_DISPLAY_COLS):
        for row in range(VISION_DISPLAY_ROWS):
            cell_rect = pygame.Rect(vision_x_offset + col * VISION_CELL_SIZE,
                                    vision_y_offset + row * VISION_CELL_SIZE,
                                    VISION_CELL_SIZE, VISION_CELL_SIZE)
            if (col, row) in visible_cells:
                pygame.draw.rect(screen, visible_cells[(col, row)], cell_rect)
            pygame.draw.rect(screen, GRAY, cell_rect, 1)

# --- Q-learning Agent ---
class QLearner():

    def read_q_values_file(self, file_path):

        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
    
            for row in reader:
                state = row[0].strip()
                values = np.array([float(row[1]), float(row[2]), float(row[3])])
            self.q_values[state] = values

    def __init__(self, 
		q_values_filepath = None, 
		epsilon=0.5, 
		learning_rate=0.1, 
		gamma=0.99):
	
        self.epsilon = epsilon
        self.q_values = defaultdict(lambda: np.zeros(3, np.float64))
        if(q_values_filepath != None):
            self.read_q_values_file(q_values_filepath)
            
        self.last_state = ''
        self.last_action = 0
        self.gamma = gamma
        self.learning_rate = learning_rate

    def _egreedy_policy(self, state):
        
        if np.random.random() < self.epsilon:
            return np.random.choice(3) 
        else:
            return np.argmax(self.q_values[state])
    
    def action(self, state):
        self.last_action = self._egreedy_policy(state)
        return self.last_action

    def update(self, next_state, reward, done): 
        td_target = reward + self.gamma * max(self.q_values[next_state])
        td_error = td_target - self.q_values[self.last_state][self.last_action]
        self.q_values[self.last_state][self.last_action] += self.learning_rate * td_error
        if done:
            print(f"{self.last_state=}")
            print(f"{reward=}")
            print(f"{max(self.q_values[next_state])=}")
            print(f"{td_target=}")
            print(f"{td_error=}")
            print(f"{next_state=}")
            print("Q-values:", self.q_values[self.last_state])
            #input("wait")
        self.last_state = next_state

    def q_values_table_str(self):
        table = []
        for key, arr in self.q_values.items():
            table.append([key] + list(arr))
        return table

def main():
    global FPS
    avg_score = 0  
    episode = 0
    snake = [
        (GRID_WIDTH // 2, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
    ]
    direction = (1, 0)
    food = random_food_position(snake)
    game_over = False
    game_over_time = None

    qlearner = QLearner(epsilon=0.1,
                        learning_rate=1,
                        gamma=0.9999)
    
    # Custom event: save Q-state every 5 seconds.
    TIMER_QSTATE_SAVE_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(TIMER_QSTATE_SAVE_EVENT, 5000)

    last_score = 0
    with open("score_history.csv","w") as f:
        f.write("episode,score\n")
    # Main loop
    ticks = 0
    score = 0
    
    
    while True:
        screen.fill(BLACK)
        
        # compute current visible state.
        visible_state, visible_cells = compute_visible_state(snake, food, direction)
        
        reward = 0
        # Use Q-learner to choose an action if game is not over.
        go = ["straight", "left", "right"]
        if not game_over:
            action = qlearner.action(visible_state)
            direction = relative_turn(direction, go[action])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == TIMER_QSTATE_SAVE_EVENT:
                with open("q_values.csv", "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["state", "straight", "left", "right"])
                    writer.writerows(qlearner.q_values_table_str())
                print("saved_q_state")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if FPS > 10:
                       FPS = 10
                    else:
                       FPS = 100000 
        # Update game state.
        head_x, head_y = snake[0]
        dx, dy = direction
        new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)
        if new_head in snake or len(snake) == 1:
            game_over = True
        else:
            snake.insert(0, new_head)
            ate_food = (new_head == food)
            if ate_food:
                # Reward for food eaten.
                reward = 10
                food = random_food_position(snake)
            else:
                reward = -0.5
                # Small time penalty.
                if(ticks == 1000):
                    reward = -1
                    snake.pop()
                    print("health - 1")
                    ticks = 0
                ticks += 1
                snake.pop()
        if game_over:
            print(f"{score=}")
            reward = -50 
            with open("score_history.csv","a") as f:
                f.write(f"{episode},{score}\n")
            episode += 1
            avg_score = (avg_score * (episode - 1) + (len(snake) - 3)) / episode
            game_over_time = pygame.time.get_ticks()
            snake = [
                (GRID_WIDTH // 2, GRID_HEIGHT // 2),
                (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
                (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
            ]
            direction = (1, 0)
            food = random_food_position(snake)
            game_over = False
            last_score = 0
            # Decay epsilon after each episode.
            #qlearner.epsilon = qlearner.epsilon * (1 - 0.001*avg_score)
        
        visible_state, visible_cells = compute_visible_state(snake, food, direction)
        draw_game_area(snake, food)
        draw_vision_area(visible_cells)
        pygame.draw.line(screen, GRAY, (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
        
        qlearner.update(visible_state, reward, True)
        score = len(snake) - 3
        score_text = score_font.render(f"episode: {episode} avg: {avg_score:.2f} score: {score} eps: {qlearner.epsilon:.2f}", True, WHITE)
        score_rect = score_text.get_rect(topright=(WINDOW_WIDTH - 10, 10))
        screen.blit(score_text, score_rect)
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
