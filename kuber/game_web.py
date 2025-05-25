import argparse
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import json
import time
import threading
import random
import numpy as np
import logging

# Default game configuration
GRID_WIDTH = 15
GRID_HEIGHT = 15
VISION_RADIUS = 5
VISION_DISPLAY_COLS = 11
VISION_DISPLAY_ROWS = 11
FPS = 10
MAX_SNAKES = 10  # Optional limit on concurrent snakes

# Flask setup
app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Global game state
GAME_OVER = False
game_over_lock = threading.Lock()

# Global food positions and snake storage
FOODS = set()
snakes = {}
snake_locks = {}

# Helper: spawn new food avoiding snakes and existing food
def spawn_food():
    occupied = {pos for game in snakes.values() for pos in game.snake} | FOODS
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in occupied:
            FOODS.add(pos)
            break

# Safe spawn location for a new snake of length 3
def find_safe_spawn_location():
    occupied = {pos for g in snakes.values() for pos in g.snake} | FOODS
    for _ in range(1000):
        head = (random.randrange(GRID_WIDTH), random.randrange(GRID_HEIGHT))
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            body = [(head[0] - i*dx, head[1] - i*dy) for i in range(3)]
            if all(0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT for x,y in body) and not any(pos in occupied for pos in body):
                return body, (dx, dy)
    # Fallback center spawn
    fallback = [(GRID_WIDTH//2 - i, GRID_HEIGHT//2) for i in range(3)]
    return fallback, (1, 0)

# End the game for all snakes
def end_game_all():
    global GAME_OVER
    with game_over_lock:
        GAME_OVER = True

# Reset the game: terminate streams, clear state, respawn food
def reset_game():
    global GAME_OVER, FOODS, snakes, snake_locks
    # Signal game over to end streams
    with game_over_lock:
        GAME_OVER = True
    time.sleep(0.1)
    # Clear all state
    with game_over_lock:
        GAME_OVER = False
    snakes.clear()
    snake_locks.clear()
    FOODS.clear()
    spawn_food()
    logging.info("Game reset: all snakes removed, food respawned.")

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
        spawn_positions, spawn_direction = find_safe_spawn_location()
        self.snake = spawn_positions
        self.direction = spawn_direction
        self.ticks = 0
        logging.info(f"Snake {self.snake_id} spawned at {self.snake[0]} dir={self.direction}")

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
        if new_head in occupied:
            end_game_all()
            return
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
    reset_game()
    while True:
        time.sleep(1.0 / FPS)
        for sid, game in list(snakes.items()):
            with snake_locks[sid]:
                game.update()

@app.route('/snake/<sid>', methods=['GET'])
def stream_vision(sid):
    if len(snakes) >= MAX_SNAKES and sid not in snakes:
        return jsonify({'error': 'server full'}), 503
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
        if GAME_OVER:
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
    threading.Thread(target=game_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

