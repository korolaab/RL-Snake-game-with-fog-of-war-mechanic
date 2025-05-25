import argparse
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import json
import time
import threading
import random
import numpy as np

# Default game configuration
GRID_WIDTH = 15
GRID_HEIGHT = 15
VISION_RADIUS = 5
VISION_DISPLAY_COLS = 11
VISION_DISPLAY_ROWS = 11
FPS = 10

# Flask setup
app = Flask(__name__)
CORS(app)

# Global game state
GAME_OVER = False
game_over_lock = threading.Lock()

# Helper to spawn new food avoiding all snakes and existing food
def spawn_food():
    occupied = {pos for game in snakes.values() for pos in game.snake}
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in occupied and pos not in FOODS:
            FOODS.add(pos)
            break

# End the game for all snakes
def end_game_all():
    global GAME_OVER
    with game_over_lock:
        GAME_OVER = True

# Reset the game
def reset_game():
    global GAME_OVER, FOODS
    with game_over_lock:
        GAME_OVER = False
    FOODS.clear()
    for sid in list(snakes.keys()):
        with snake_locks[sid]:
            snakes[sid].reset()
    spawn_food()

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Snake Vision Stream API configuration")
    parser.add_argument("--grid_width", type=int, default=GRID_WIDTH)
    parser.add_argument("--grid_height", type=int, default=GRID_HEIGHT)
    parser.add_argument("--vision_radius", type=int, default=VISION_RADIUS)
    parser.add_argument("--vision_display_cols", type=int, default=VISION_DISPLAY_COLS)
    parser.add_argument("--vision_display_rows", type=int, default=VISION_DISPLAY_ROWS)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

# Seed RNGs
def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

class SnakeGame:
    def __init__(self, snake_id):
        self.snake_id = snake_id
        self.direction = (1, 0)
        self.snake = []
        self.ticks = 0
        self.reset()

    def reset(self):
        self.snake = [
            (GRID_WIDTH // 2, GRID_HEIGHT // 2),
            (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
            (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
        ]
        self.direction = (1, 0)
        self.ticks = 0

    def relative_turn(self, cmd):
        if cmd == 'left':
            return (self.direction[1], -self.direction[0])
        if cmd == 'right':
            return (-self.direction[1], self.direction[0])
        return self.direction

    def turn(self, cmd):
        self.direction = self.relative_turn(cmd)

    def update(self):
        global GAME_OVER
        with game_over_lock:
            if GAME_OVER:
                return
        
        head = self.snake[0]
        new_head = ((head[0] + self.direction[0]) % GRID_WIDTH,
                    (head[1] + self.direction[1]) % GRID_HEIGHT)
        occupied = {pos for game in snakes.values() for pos in game.snake}
        # Collision: with self or others
        if new_head in occupied:
            end_game_all()
            return
        # Move
        self.snake.insert(0, new_head)
        if new_head in FOODS:
            FOODS.remove(new_head)
            spawn_food()
        else:
            self.snake.pop()
        self.ticks += 1

    def get_visible_cells(self):
        head = self.snake[0]
        rotate_map = {
            (0, -1): lambda dx, dy: (dx, dy),
            (1, 0):  lambda dx, dy: (dy, -dx),
            (0, 1):  lambda dx, dy: (-dx, -dy),
            (-1, 0): lambda dx, dy: (-dy, dx)
        }
        rotate = rotate_map.get(self.direction, rotate_map[(0, -1)])
        other_heads = {g.snake[0] for sid, g in snakes.items() if sid != self.snake_id}
        other_bodies = {pos for sid, g in snakes.items() if sid != self.snake_id for pos in g.snake[1:]}
        vis = {}
        for dx in range(-VISION_RADIUS, VISION_RADIUS + 1):
            for dy in range(-VISION_RADIUS, VISION_RADIUS + 1):
                if abs(dx) + abs(dy) > VISION_RADIUS:
                    continue
                rx, ry = rotate(dx, dy)
                cx = VISION_DISPLAY_COLS // 2 + rx
                cy = VISION_DISPLAY_ROWS // 2 + ry
                if not (0 <= cx < VISION_DISPLAY_COLS and 0 <= cy < VISION_DISPLAY_ROWS):
                    continue
                px = (head[0] + dx) % GRID_WIDTH
                py = (head[1] + dy) % GRID_HEIGHT
                pos = (px, py)
                if pos == head:
                    obj = 'HEAD'
                elif pos in self.snake[1:]:
                    obj = 'BODY'
                elif pos in other_heads:
                    obj = 'OTHER_HEAD'
                elif pos in other_bodies:
                    obj = 'OTHER_BODY'
                elif pos in FOODS:
                    obj = 'FOOD'
                else:
                    obj = 'EMPTY'
                vis[f"{cx},{cy}"] = obj
        return vis


# Background game loop thread
def game_loop():
    while True:
        time.sleep(1.0 / FPS)
        for sid, game in list(snakes.items()):
            with snake_locks[sid]:
                game.update()

@app.route('/snake/<sid>', methods=['GET'])
def stream_vision(sid):
    if sid not in snakes:
        snakes[sid] = SnakeGame(sid)
        snake_locks[sid] = threading.Lock()
    
    def gen():
        while True:
            with game_over_lock:
                is_game_over = GAME_OVER
            
            with snake_locks[sid]:
                vis = snakes[sid].get_visible_cells()
            
            yield json.dumps({'snake_id': sid, 'visible_cells': vis, 'game_over': is_game_over}) + '\n'
            
            if is_game_over:
                break
                
            time.sleep(1.0 / FPS)
    
    return Response(gen(), mimetype='application/x-ndjson', headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})

@app.route('/snake/<sid>/move', methods=['POST'])
def move_snake(sid):
    if sid not in snakes:
        return jsonify({'error': 'not found'}), 404
    
    with game_over_lock:
        is_game_over = GAME_OVER
    
    if is_game_over:
        return jsonify({'snake_id': sid, 'game_over': True})
    
    data = request.get_json(force=True)
    cmd = data.get('move')
    if cmd not in ('left', 'right'):
        return jsonify({'error': 'Invalid move'}), 400
    
    with snake_locks[sid]:
        snakes[sid].turn(cmd)
    
    return jsonify({'snake_id': sid, 'game_over': False})

@app.route('/state', methods=['GET'])
def state():
    grid = {f"{x},{y}": [] for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT)}
    for sid, game in snakes.items():
        with snake_locks[sid]:
            for i, p in enumerate(game.snake):
                cell = f"{p[0]},{p[1]}"
                typ = 'HEAD' if i == 0 else 'BODY'
                grid[cell].append({'type': typ, 'snake_id': sid})
    for food in FOODS:
        cell = f"{food[0]},{food[1]}"
        grid[cell].append({'type': 'FOOD', 'snake_id': None})
    for cell, v in grid.items():
        if not v:
            grid[cell] = [{'type': 'EMPTY'}]
    
    visions = {sid: game.get_visible_cells() for sid, game in snakes.items()}
    
    with game_over_lock:
        global_game_over = GAME_OVER
    
    statuses = {sid: global_game_over for sid in snakes.keys()}
    
    return jsonify({'grid': grid, 'visions': visions, 'status': statuses, 'global_game_over': global_game_over})

@app.route('/reset', methods=['POST'])
def reset():
    reset_game()
    return jsonify({'message': 'Game reset successfully'})

@app.route('/', methods=['GET'])
def home():
    with game_over_lock:
        is_game_over = GAME_OVER
    return jsonify({'message': 'Snake Vision Stream API', 'game_over': is_game_over})

if __name__ == '__main__':
    args = parse_args()
    GRID_WIDTH = args.grid_width
    GRID_HEIGHT = args.grid_height
    VISION_RADIUS = args.vision_radius
    VISION_DISPLAY_COLS = args.vision_display_cols
    VISION_DISPLAY_ROWS = args.vision_display_rows
    FPS = args.fps
    set_seed(args.seed)
    print(f"Config: {GRID_WIDTH}x{GRID_HEIGHT}, R={VISION_RADIUS}, D={VISION_DISPLAY_COLS}x{VISION_DISPLAY_ROWS}, FPS={FPS}, seed={args.seed}")

    # Global food positions
    FOODS = set()

    # Global snake storage and locks
    snakes = {}
    snake_locks = {}

    # Initialize first food
    spawn_food()

    threading.Thread(target=game_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
