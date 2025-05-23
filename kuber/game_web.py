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

# Parse command-line arguments for configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Snake Vision Stream API configuration")
    parser.add_argument("--grid_width", type=int, default=GRID_WIDTH, help="Width of the game grid")
    parser.add_argument("--grid_height", type=int, default=GRID_HEIGHT, help="Height of the game grid")
    parser.add_argument("--vision_radius", type=int, default=VISION_RADIUS, help="Radius of vision around the snake head")
    parser.add_argument("--vision_display_cols", type=int, default=VISION_DISPLAY_COLS, help="Number of columns in the vision display")
    parser.add_argument("--vision_display_rows", type=int, default=VISION_DISPLAY_ROWS, help="Number of rows in the vision display")
    parser.add_argument("--fps", type=int, default=FPS, help="Frames (ticks) per second for the stream")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()

# Set random seed if provided
def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

class SnakeGame:
    def __init__(self, snake_id):
        self.snake_id = snake_id
        self.is_active = True
        self.game_over = False
        self.ticks = 0
        self.last_action = 0
        self.reset()

    def reset(self):
        self.snake = [
            (GRID_WIDTH // 2, GRID_HEIGHT // 2),
            (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
            (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
        ]
        self.direction = (1, 0)
        self.food = self.random_food_position()
        self.game_over = False
        self.ticks = 0

    def random_food_position(self):
        while True:
            pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            occupied = set(p for s in snakes.values() for p in s.snake)
            if pos not in occupied:
                return pos

    def relative_turn(self, turn_command):
        if turn_command == "left":
            return (self.direction[1], -self.direction[0])
        elif turn_command == "right":
            return (-self.direction[1], self.direction[0])
        return self.direction

    def update(self, move=None):
        if self.game_over:
            return
        if move:
            self.direction = self.relative_turn(move)
            self.last_action = 1 if move in ["left", "right"] else 0
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)
        occupied = set(p for s in snakes.values() for p in s.snake)
        if new_head in occupied:
            self.game_over = True
            return
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.food = self.random_food_position()
        else:
            if self.ticks >= 50:
                self.ticks = 0
            else:
                self.snake.pop()
        self.ticks += 1

    def get_visible_cells(self):
        head_x, head_y = self.snake[0]
        visible_cells = {}
        if self.direction == (0, -1):
            rotate = lambda dx, dy: (dx, dy)
        elif self.direction == (1, 0):
            rotate = lambda dx, dy: (dy, -dx)
        elif self.direction == (0, 1):
            rotate = lambda dx, dy: (-dx, -dy)
        elif self.direction == (-1, 0):
            rotate = lambda dx, dy: (-dy, dx)
        else:
            rotate = lambda dx, dy: (dx, dy)
        other_heads = [s.snake[0] for sid, s in snakes.items() if sid != self.snake_id]
        other_bodies = {pos for sid, s in snakes.items() if sid != self.snake_id for pos in s.snake[1:]}
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
                pos = (global_x, global_y)
                if pos == self.snake[0]: obj = 'HEAD'
                elif pos in self.snake[1:]: obj = 'BODY'
                elif pos in other_heads: obj = 'OTHER_HEAD'
                elif pos in other_bodies: obj = 'OTHER_BODY'
                elif pos == self.food: obj = 'FOOD'
                else: obj = 'EMPTY'
                visible_cells[f"{disp_col},{disp_row}"] = obj
        return visible_cells

# Global storage
snakes = {}
snake_locks = {}
app = Flask(__name__)
CORS(app)

@app.route('/snake/<snake_id>', methods=['GET'])
def stream_snake_vision(snake_id):
    if snake_id not in snakes:
        snakes[snake_id] = SnakeGame(snake_id)
        snake_locks[snake_id] = threading.Lock()
    def generate_vision_stream():
        snake = snakes[snake_id]
        while snake.is_active and not snake.game_over:
            try:
                with snake_locks[snake_id]:
                    snake.update()
                    visible_cells = snake.get_visible_cells()
                data = {'snake_id': snake_id, 'visible_cells': visible_cells}
                yield json.dumps(data) + '\n'
                if snake.game_over:
                    yield json.dumps({'snake_id': snake_id, 'game_over': True}) + '\n'
                    break
                time.sleep(1.0 / FPS)
            except Exception as e:
                yield json.dumps({'snake_id': snake_id, 'error': str(e)}) + '\n'
                break
    return Response(generate_vision_stream(), mimetype='application/x-ndjson', headers={'Cache-Control': 'no-cache','Connection': 'keep-alive'})

@app.route('/snake/<snake_id>/move', methods=['POST'])
def control_snake(snake_id):
    if snake_id not in snakes:
        return jsonify({'error': 'Snake not found'}), 404
    data = request.get_json()
    if not data or 'move' not in data:
        return jsonify({'error': 'Move is required'}), 400
    move = data['move']
    if move not in ['left', 'right']:
        return jsonify({'error': 'Invalid move. Use: left or right'}), 400
    with snake_locks[snake_id]:
        snakes[snake_id].update(move)
    return jsonify({'snake_id': snake_id, 'move_applied': move, 'game_over': snakes[snake_id].game_over})

@app.route('/snake/<snake_id>/reset', methods=['POST'])
def reset_snake(snake_id):
    if snake_id not in snakes:
        snakes[snake_id] = SnakeGame(snake_id)
        snake_locks[snake_id] = threading.Lock()
    else:
        with snake_locks[snake_id]:
            snakes[snake_id].reset()
    return jsonify({'snake_id': snake_id, 'message': 'Snake reset successfully'})

@app.route('/snakes', methods=['GET'])
def list_snakes():
    return jsonify({'snakes': [{'snake_id': sid, 'is_active': snake.is_active, 'game_over': snake.game_over, 'length': len(snake.snake)} for sid, snake in snakes.items()]})

@app.route('/state', methods=['GET'])
def game_state():
    grid = {f"{x},{y}": [] for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT)}
    for sid, snake in snakes.items():
        with snake_locks[sid]:
            for idx, segment in enumerate(snake.snake):
                cell = f"{segment[0]},{segment[1]}"
                typ = 'HEAD' if idx == 0 else 'BODY'
                grid[cell].append({'type': typ, 'snake_id': sid})
            food_cell = f"{snake.food[0]},{snake.food[1]}"
            grid[food_cell].append({'type': 'FOOD', 'snake_id': sid})
    for cell, items in grid.items():
        if not items:
            grid[cell] = [{'type': 'EMPTY'}]
    visions = {sid: snakes[sid].get_visible_cells() for sid in snakes}
    return jsonify({'grid': grid, 'visions': visions})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Snake Vision Stream API',
        'usage': {
            'stream_vision': 'GET /snake/{snake_id}',
            'control_snake': 'POST /snake/{snake_id}/move with {"move": "left|right"}',
            'reset_snake': 'POST /snake/{snake_id}/reset',
            'list_snakes': 'GET /snakes',
            'game_state': 'GET /state'
        },
        'example': {
            'stream': 'curl http://localhost:5000/snake/my_snake_01',
            'move': """curl -X POST http://localhost:5000/snake/my_snake_01/move -H 'Content-Type: application/json' -d '{\\"move\\": \\\"left\\\"}'""",
            'state': 'curl http://localhost:5000/state'
        },
        'output_format': {
            'snake_id': 'string',
            'visible_cells': {'x,y': 'object_type'},
            'grid': {'x,y': [{'type': 'EMPTY|HEAD|BODY|FOOD', 'snake_id': 'string'}]}
        }
    })

if __name__ == '__main__':
    args = parse_args()
    GRID_WIDTH = args.grid_width
    GRID_HEIGHT = args.grid_height
    VISION_RADIUS = args.vision_radius
    VISION_DISPLAY_COLS = args.vision_display_cols
    VISION_DISPLAY_ROWS = args.vision_display_rows
    FPS = args.fps
    set_seed(args.seed)

    print("Starting Snake Vision Stream API with configuration:")
    print(f" Grid: {GRID_WIDTH}x{GRID_HEIGHT}, Vision Radius: {VISION_RADIUS}, Display: {VISION_DISPLAY_COLS}x{VISION_DISPLAY_ROWS}, FPS: {FPS}, Seed: {args.seed}")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

