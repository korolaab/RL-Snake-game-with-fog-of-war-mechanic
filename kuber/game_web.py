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
    parser.add_argument("--fps", type=int, default=FPS, help="Ticks per second for game updates and streaming")
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
        self.direction = (1,0)
        self.food = None
        self.reset()

    def reset(self):
        self.snake = [
            (GRID_WIDTH//2, GRID_HEIGHT//2),
            (GRID_WIDTH//2-1, GRID_HEIGHT//2),
            (GRID_WIDTH//2-2, GRID_HEIGHT//2)
        ]
        self.direction = (1,0)
        self.food = self.random_food_position()
        self.game_over = False
        self.ticks = 0

    def random_food_position(self):
        occupied = set(pos for game in snakes.values() for pos in game.snake)
        while True:
            pos = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            if pos not in occupied:
                return pos

    def relative_turn(self, cmd):
        if cmd=='left': return (self.direction[1], -self.direction[0])
        if cmd=='right': return (-self.direction[1], self.direction[0])
        return self.direction

    def update(self):
        if self.game_over: return
        # Move head
        head = self.snake[0]
        new_head = ((head[0]+self.direction[0])%GRID_WIDTH, (head[1]+self.direction[1])%GRID_HEIGHT)
        occupied = set(pos for game in snakes.values() for pos in game.snake)
        if new_head in occupied:
            self.game_over=True
            return
        self.snake.insert(0, new_head)
        # Eat or move tail
        if new_head==self.food:
            self.food=self.random_food_position()
        else:
            self.snake.pop()

    def turn(self, cmd):
        self.direction = self.relative_turn(cmd)

    def get_visible_cells(self):
        head = self.snake[0]
        dirs = { (0,-1):lambda dx,dy:(dx,dy), (1,0):lambda dx,dy:(dy,-dx),
                 (0,1):lambda dx,dy:(-dx,-dy), (-1,0):lambda dx,dy:(-dy,dx) }
        rotate = dirs.get(self.direction, dirs[(0,-1)])
        others_heads = {g.snake[0] for sid,g in snakes.items() if sid!=self.snake_id}
        others_body = {pos for sid,g in snakes.items() if sid!=self.snake_id for pos in g.snake[1:]}
        vis={}  
        for dx in range(-VISION_RADIUS, VISION_RADIUS+1):
            for dy in range(-VISION_RADIUS, VISION_RADIUS+1):
                if abs(dx)+abs(dy)>VISION_RADIUS: continue
                rx,ry=rotate(dx,dy)
                cx,cy=(VISION_DISPLAY_COLS//2+rx, VISION_DISPLAY_ROWS//2+ry)
                if not(0<=cx<VISION_DISPLAY_COLS and 0<=cy<VISION_DISPLAY_ROWS): continue
                px,py=(head[0]+dx)%GRID_WIDTH, (head[1]+dy)%GRID_HEIGHT
                pos=(px,py)
                if pos==head: o='HEAD'
                elif pos in self.snake[1:]: o='BODY'
                elif pos in others_heads: o='OTHER_HEAD'
                elif pos in others_body: o='OTHER_BODY'
                elif pos==self.food: o='FOOD'
                else: o='EMPTY'
                vis[f"{cx},{cy}"]=o
        return vis

# Global
snakes={}  
snake_locks={}
app=Flask(__name__)
CORS(app)

# Background loop
def game_loop():
    while True:
        time.sleep(1.0/FPS)
        for sid,game in snakes.items():
            with snake_locks[sid]:
                game.update()
threading.Thread(target=game_loop, daemon=True).start()

@app.route('/snake/<sid>', methods=['GET'])
def stream_vision(sid):
    if sid not in snakes:
        snakes[sid]=SnakeGame(sid); snake_locks[sid]=threading.Lock()
    def gen():
        while True:
            with snake_locks[sid]:
                vis=snakes[sid].get_visible_cells()
                over=snakes[sid].game_over
            yield json.dumps({'snake_id':sid,'visible_cells':vis,'game_over':over})+'\n'
            if over: break
            time.sleep(1.0/FPS)
    return Response(gen(), mimetype='application/x-ndjson', headers={'Cache-Control':'no-cache','Connection':'keep-alive'})

@app.route('/snake/<sid>/move', methods=['POST'])
def move_snake(sid):
    if sid not in snakes: return jsonify({'error':'not found'}),404
    data=request.get_json(force=True)
    cmd=data.get('move')
    if cmd not in('left','right'): return jsonify({'error':'Invalid move'}),400
    with snake_locks[sid]: snakes[sid].turn(cmd)
    return jsonify({'snake_id':sid,'game_over':snakes[sid].game_over})

@app.route('/state', methods=['GET'])
def state():
    grid={f"{x},{y}":[] for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT)}
    for sid,game in snakes.items():
        with snake_locks[sid]:
            for i,p in enumerate(game.snake):
                grid[f"{p[0]},{p[1]}"] .append({'type':'HEAD' if i==0 else 'BODY','snake_id':sid})
            grid[f"{game.food[0]},{game.food[1]}"] .append({'type':'FOOD','snake_id':sid})
    for c,v in grid.items():
        if not v: grid[c]=[{'type':'EMPTY'}]
    vis={sid: game.get_visible_cells() for sid,game in snakes.items()}
    return jsonify({'grid':grid,'visions':vis})

if __name__=='__main__':
    args=parse_args()
    GRID_WIDTH,GRID_HEIGHT=args.grid_width,args.grid_height
    VISION_RADIUS, VISION_DISPLAY_COLS, VISION_DISPLAY_ROWS = args.vision_radius, args.vision_display_cols, args.vision_display_rows
    FPS=args.fps; set_seed(args.seed)
    print(f"Config: {GRID_WIDTH}x{GRID_HEIGHT}, R={VISION_RADIUS}, D={VISION_DISPLAY_COLS}x{VISION_DISPLAY_ROWS}, FPS={FPS}, seed={args.seed}")
    app.run(host='0.0.0.0',port=5000,debug=True,threaded=True)

