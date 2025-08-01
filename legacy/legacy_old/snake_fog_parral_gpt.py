import threading
from queue import Queue
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
FPS = 100000

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

# --- Global Shared Statistics ---
shared_stats = {
    'episodes': 0,
    'total_score': 0
}
stats_lock = threading.Lock()

# --- Q-Learner with Thread-Safety ---
class QLearner:
    def __init__(self, epsilon=0.2, learning_rate=0.1, gamma=0.95):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_values = defaultdict(lambda: np.zeros(3, np.float64))
        self.lock = threading.Lock()

    def select_action(self, state):
        with self.lock:
            if random.random() < self.epsilon:
                return random.choice([0, 1, 2])
            else:
                return int(np.argmax(self.q_values[state]))
    
    def update(self, state, action, reward, next_state, done):
        with self.lock:
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
        with self.lock:
            with open("q_values.csv", "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["state", "straight", "left", "right"])
                writer.writerows(self.q_values_table_str())

# --- Headless Snake Simulation ---
def simulate_game(thread_id, qlearner, exp_queue):
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

    # A simple state representation.
    #def compute_state(snake, food, direction):
    #    head = snake[0]
    #    return f"{head}_{food}_{direction}"

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
                    c = 'H'
                elif cell in snake:
                    c = 'S'
                elif cell == food:
                    c = 'A'
                else:
                    c = '_'
                
                state_str += c
        return state_str

    state = compute_visible_state(snake, food, direction)
    
    # Run forever.
    while True:
        # Choose action from shared Q-learner.
        action = qlearner.select_action(state)
        # Actions: 0: straight, 1: left, 2: right.
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
        new_head = ((head[0] + direction[0]) % grid_width,
                    (head[1] + direction[1]) % grid_height)
        done = False
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

        next_state = compute_visible_state(snake, food, direction)
        exp_queue.put((state, action, reward, next_state, done))
        state = next_state

        if done:
            # Update shared statistics.
            with stats_lock:
                shared_stats['episodes'] += 1
                shared_stats['total_score'] += (len(snake) - 3)
            # Reset game state.
            snake = [
                (grid_width // 2, grid_height // 2),
                (grid_width // 2 - 1, grid_height // 2),
                (grid_width // 2 - 2, grid_height // 2)
            ]
            direction = (1, 0)
            food = random_food(snake)
            state = compute_visible_state(snake, food, direction)
        #time.sleep(0.001)  # tiny delay to yield control

# --- Updater Thread: Consumes Experiences and Updates Q-Learner ---
def updater_thread(qlearner, exp_queue):
    while True:
        transition = exp_queue.get()  # blocking
        if transition is None:
            break  # exit signal
        state, action, reward, next_state, done = transition
        qlearner.update(state, action, reward, next_state, done)
        exp_queue.task_done()

# --- Stats Monitor Thread: Displays Episodes and Average Score ---
def stats_monitor():
    last_episodes = 0
    last_score = 0
    while True:
        time.sleep(1)  # update every 1 seconds
        with stats_lock:
            episodes = shared_stats['episodes']
            total_score = shared_stats['total_score']
            shared_stats['total_score'] = 0
        episodes_per_second = episodes - last_episodes
        avg_score =  total_score / episodes_per_second if episodes > 0 else 0
	
        print(f"Total episodes: {episodes}, Average score: {avg_score:.2f}, {episodes_per_second}ep/s")
        last_episodes = episodes

# --- Main function to start simulations and threads ---
def main():
    num_simulations = 1 # number of parallel game threads
    exp_queue = Queue()
    qlearner = QLearner(epsilon=0.1, learning_rate=1, gamma=0.99)
    
    # Start updater thread.
    updater = threading.Thread(target=updater_thread, args=(qlearner, exp_queue), daemon=True)
    updater.start()
    
    # Start stats monitor thread.
    monitor = threading.Thread(target=stats_monitor, daemon=True)
    monitor.start()
    
    # Start simulation threads.
    sim_threads = []
    for i in range(num_simulations):
        t = threading.Thread(target=simulate_game, args=(i, qlearner, exp_queue), daemon=True)
        sim_threads.append(t)
        t.start()
    
    # Run forever. (Press Ctrl+C to exit.)
    try:
        while True:
            time.sleep(5)
            qlearner.save_qvalues()
            print("save_qvalues")
    except KeyboardInterrupt:
        print("Exiting...")
        exp_queue.put(None)  # signal updater to exit
        updater.join()

if __name__ == "__main__":
    main()

