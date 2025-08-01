import random
import time
import numpy as np
import csv
from collections import defaultdict

# --- Game settings ---
CELL_SIZE = 20
GRID_WIDTH = 15
GRID_HEIGHT = 15
GAME_WIDTH = GRID_WIDTH * CELL_SIZE   # 300 pixels
GAME_HEIGHT = GRID_HEIGHT * CELL_SIZE   # 300 pixels
FPS = 100000  # (Not used in headless simulation)

# --- Snake vision settings ---
VISION_RADIUS = 5
VISION_DISPLAY_COLS = 11  
VISION_DISPLAY_ROWS = 11   
VISION_CELL_SIZE = 20     
VISION_WIDTH = VISION_DISPLAY_COLS * VISION_CELL_SIZE   # 220 pixels
VISION_HEIGHT = VISION_DISPLAY_ROWS * VISION_CELL_SIZE    # 220 pixels

# --- Window settings ---
WINDOW_WIDTH = GAME_WIDTH + VISION_WIDTH  # 300 + 220 = 520 pixels
WINDOW_HEIGHT = GAME_HEIGHT               # 300 pixels

# --- Global Statistics ---
global_episodes = 0
global_total_score = 0

# --- Q-Learner ---
class QLearner:
    def __init__(self, epsilon=0.1, learning_rate=1, gamma=0.99):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_values = defaultdict(lambda: np.zeros(3, np.float64))

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])
        else:
            return int(np.argmax(self.q_values[state]))
    
    def update(self, state, action, reward, next_state, done):
        q_next = 0 if done else max(self.q_values[next_state])
        td_target = reward + self.gamma * q_next
        td_error = td_target - self.q_values[state][action]
        self.q_values[state][action] += self.learning_rate * td_error

    def q_values_table_str(self):
        table = []
        for key, arr in self.q_values.items():
            table.append([key] + list(arr))
        return table

    def save_qvalues(self):
        with open("q_values.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["state", "straight", "left", "right"])
            writer.writerows(self.q_values_table_str())

# --- Compute a simple state representation based on the snake's vision ---
def compute_visible_state(snake, food, direction):
    head_x, head_y = snake[0]
    state_str = ""
    
    # Rotate coordinates so that "forward" is always in the same orientation.
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
    
    for dx in range(-VISION_RADIUS, VISION_RADIUS + 1):
        for dy in range(-VISION_RADIUS, VISION_RADIUS + 1):
            if abs(dx) + abs(dy) > VISION_RADIUS:
                continue
            # We simply build a string of characters representing each cell in the vision diamond.
            global_x = (head_x + dx) % GRID_WIDTH
            global_y = (head_y + dy) % GRID_HEIGHT
            cell = (global_x, global_y)
            if (dx, dy) == (0, 0):
                c = 'H'
            elif cell in snake:
                c = 'S'
            elif cell == food:
                c = 'A'
            else:
                c = '_'
            state_str += c
    return state_str
def extract_features(snake, food, direction):
    """
    Extract features from the snake’s first-person view.
    Features include:
      - Relative food position (rx, ry) in the rotated coordinate system (normalized).
      - Danger flags for the immediate front, left, and right cells.
    The rotated coordinate system is defined such that "forward" is always up.
    """
    head = snake[0]

    # Define a rotation function to align "forward" (relative to snake's direction) to up.
    if direction == (0, -1):  # up: no rotation
        def rotate(dx, dy): 
            return (dx, dy)
    elif direction == (1, 0):  # right: rotate 90° clockwise
        def rotate(dx, dy): 
            return (dy, -dx)
    elif direction == (0, 1):  # down: rotate 180°
        def rotate(dx, dy): 
            return (-dx, -dy)
    elif direction == (-1, 0):  # left: rotate 90° counterclockwise
        def rotate(dx, dy): 
            return (-dy, dx)
    else:
        def rotate(dx, dy):
            return (dx, dy)

    # Compute relative food position (in global coordinates)
    food_dx = food[0] - head[0]
    food_dy = food[1] - head[1]
    # Rotate so that "forward" is up
    food_rx, food_ry = rotate(food_dx, food_dy)
    # Normalize by grid dimensions (optional – adjust as needed)
    norm_food_rx = food_rx 
    norm_food_ry = food_ry

    # Define unit vectors for front, left, and right in the rotated system.
    # In the rotated system, "forward" is up (i.e. (0, -1))
    front = (0, -1)
    left  = (-1, 0)
    right = (1, 0)
    
    # Compute danger flags. Since our grid wraps around, we consider a cell dangerous if it is part of the snake body.
    def is_danger(relative_direction):
        # The adjacent cell in the given relative direction
        cell = ((head[0] + relative_direction[0]) % GRID_WIDTH,
                (head[1] + relative_direction[1]) % GRID_HEIGHT)
        # We consider the cell dangerous if it is occupied by any part of the snake (except the head).
        return 1 if cell in snake[1:] else 0

    # However, the danger should be computed in the rotated system.
    # For that, we rotate the unit vectors for front, left, and right.
    front_rot = rotate(*front)
    left_rot  = rotate(*left)
    right_rot = rotate(*right)
    
    # Now, compute danger using these rotated directions.
    # Note: here, since the snake's body is stored in global coordinates,
    # we convert the rotated direction back to a global direction.
    # A simple way is to assume that a small rotation does not change the global cell
    # if we are only checking immediate neighbors.
    danger_front = 1 if ((head[0] + front[0]) % GRID_WIDTH, (head[1] + front[1]) % GRID_HEIGHT) in snake[1:] else 0
    danger_left  = 1 if ((head[0] + left[0]) % GRID_WIDTH, (head[1] + left[1]) % GRID_HEIGHT) in snake[1:] else 0
    danger_right = 1 if ((head[0] + right[0]) % GRID_WIDTH, (head[1] + right[1]) % GRID_HEIGHT) in snake[1:] else 0

    # Combine features into a vector.
    features = [norm_food_rx, norm_food_ry, danger_front, danger_left, danger_right]
    return features
def features_to_str(features):
    """
    Converts a list of features into a string.
    Each feature is rounded to 2 decimal places and joined with underscores.
    """
    return '_'.join(str(round(f, 2)) for f in features)

# --- Simulate one episode (game until collision) ---
def simulate_episode(qlearner):
    grid_width, grid_height = 15, 15

    # Initialize game state.
    snake = [
        (grid_width // 2, grid_height // 2),
        (grid_width // 2 - 1, grid_height // 2),
        (grid_width // 2 - 2, grid_height // 2)
    ]
    direction = (1, 0)  # moving right

    def random_food(snake):
        while True:
            pos = (random.randint(0, grid_width - 1), random.randint(0, grid_height - 1))
            if pos not in snake:
                return pos
    food = random_food(snake)

    #state = compute_visible_state(snake, food, direction)
    state_features = extract_features(snake, food, direction)
    state = features_to_str(state_features)
    done = False
    while not done:
        action = qlearner.select_action(state)
        # Relative turning: 0: straight, 1: left, 2: right.
        def relative_turn(direction, turn_command):
            if turn_command == "left":
                return (direction[1], -direction[0])
            elif turn_command == "right":
                return (-direction[1], direction[0])
            else:
                return direction
        moves = ["straight", "left", "right"]
        direction = relative_turn(direction, moves[action])
        head = snake[0]
        new_head = ((head[0] + direction[0]) % grid_width, (head[1] + direction[1]) % grid_height)
        reward = -0.1  # time penalty
        if new_head in snake:
            done = True
            reward = -50  # collision penalty
        else:
            snake.insert(0, new_head)
            if new_head == food:
                reward = 10  # reward for eating food
                food = random_food(snake)
            else:
                snake.pop()
        #next_state = compute_visible_state(snake, food, direction)
        state_features = extract_features(snake, food, direction)
        next_state = features_to_str(state_features)
        qlearner.update(state, action, reward, next_state, done)
        state = next_state
    # Score is defined as the number of apples eaten = len(snake) - 3.
    return len(snake) - 3

# --- Main loop ---
def main():
    qlearner = QLearner(epsilon=0.5, learning_rate=1, gamma=0.99)
    global global_episodes, global_total_score
    last_save_time = time.time()
    last_print_time = time.time()
    last_episodes = 0
    
    while True:
        score = simulate_episode(qlearner)
        global_episodes += 1
        global_total_score += score

        # Print stats every second.
        if time.time() - last_print_time >= 1:
            episodes_per_second = global_episodes - last_episodes
            avg_score =  global_total_score / episodes_per_second
            global_total_score = 0
	
            print(f"Total episodes: {global_episodes}, Average score: {avg_score:.2f}, {episodes_per_second}ep/s")
            last_episodes = global_episodes
            last_print_time = time.time()

        # Save Q-values every 5 seconds.
        if time.time() - last_save_time >= 5:
            qlearner.save_qvalues()
            print("Saved Q-values.")
            last_save_time = time.time()

if __name__ == "__main__":
    main()

