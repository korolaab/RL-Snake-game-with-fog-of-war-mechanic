# game/manager.py

import threading
import random
import logging
from utils.seed import set_seed  

from .snake import SnakeGame

class GameManager:
    def __init__(self, grid_width, 
                       grid_height, 
                       vision_radius, 
                       vision_display_cols, 
                       vision_display_rows, 
                       fps, 
                       seed,
                       maxStepsWithoutApple,
                       reward_config,
                       max_snakes=10):
        self.GRID_WIDTH = grid_width
        self.GRID_HEIGHT = grid_height
        self.VISION_RADIUS = vision_radius
        self.VISION_DISPLAY_COLS = vision_display_cols
        self.VISION_DISPLAY_ROWS = vision_display_rows
        self.FPS = fps
        self.MAX_SNAKES = max_snakes
        self.FOODS = set()
        self.snakes = {}
        self.snake_locks = {}
        self.GAME_OVER = False
        self.game_over_lock = threading.Lock()
        self.seed = seed 
        self.reward_config = reward_config
        self.maxStepsWithoutApple = maxStepsWithoutApple
        self.episode_number = 0  # New: episode counter
        self.frame_number = 0    # New: frame counter
        set_seed(self.seed)
        self.game_over_raised = False
        threading.Thread(target=self.game_loop, daemon=True).start()

    def state(self):
        grid = {f"{x},{y}": [] for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)}
        for sid, game in self.snakes.items():
            with self.snake_locks[sid]:
                for i, p in enumerate(game.snake):
                    cell = f"{p[0]},{p[1]}"
                    typ = 'HEAD' if i == 0 else 'BODY'
                    grid[cell].append({'type': typ, 'snake_id': sid})
        for food in self.FOODS:
            cell = f"{food[0]},{food[1]}"
            grid[cell].append({'type': 'FOOD', 'snake_id': None})
        for cell, v in grid.items():
            if not v:
                grid[cell] = [{'type': 'EMPTY'}]
        visions = {sid: game.get_visible_cells() for sid, game in self.snakes.items()}
        with self.game_over_lock:
            global_game_over = self.GAME_OVER
        statuses = {sid: global_game_over for sid in self.snakes.keys()}
        return grid, visions, statuses, self.GAME_OVER

    def spawn_food(self):
        occupied = {pos for game in self.snakes.values() for pos in game.snake} | self.FOODS
        while True:
            pos = (random.randint(0, self.GRID_WIDTH - 1), random.randint(0, self.GRID_HEIGHT - 1))
            if pos not in occupied:
                self.FOODS.add(pos)
                break

    def find_safe_spawn_location(self):
        occupied = {pos for g in self.snakes.values() for pos in g.snake} | self.FOODS
        for _ in range(1000):
            head = (random.randrange(self.GRID_WIDTH), random.randrange(self.GRID_HEIGHT))
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                body = [(head[0] - i*dx, head[1] - i*dy) for i in range(3)]
                if all(0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT for x,y in body) and not any(pos in occupied for pos in body):
                    return body, (dx, dy)
        # fallback
        fallback = [(self.GRID_WIDTH//2 - i, self.GRID_HEIGHT//2) for i in range(3)]
        return fallback, (1, 0)

    def end_game_all(self):
        with self.game_over_lock:
            self.GAME_OVER = True

    def reset_game(self):
        with self.game_over_lock:
            self.GAME_OVER = True
        import time
        time.sleep(0.1)
        with self.game_over_lock:
            self.GAME_OVER = False
        self.snakes.clear()
        self.snake_locks.clear()
        self.FOODS.clear()
        self.spawn_food()
        self.game_over_raised = False
        self.episode_number += 1
        self.frame_number = 0
        logging.info({"event": "game_reset", 
                      "action": "all_snakes_removed_food_respawned", 
                      "episode": self.episode_number, 
                      "frame": self.frame_number})

    def add_snake(self, snake_id):
        if len(self.snakes) >= self.MAX_SNAKES:
            return False
        snake = SnakeGame(snake_id, self)
        self.snakes[snake_id] = snake
        self.snake_locks[snake_id] = threading.Lock()
        return True

    def remove_snake(self, snake_id):
        if snake_id in self.snakes:
            del self.snakes[snake_id]
        if snake_id in self.snake_locks:
            del self.snake_locks[snake_id]

    def get_snake(self, snake_id):
        return self.snakes.get(snake_id)

    def get_lock(self, snake_id):
        return self.snake_locks.get(snake_id)

    def game_loop(self):
        import time
        self.reset_game()
        while True:
            time.sleep(1.0 / self.FPS)
            self.frame_number += 1
            for sid, game in list(self.snakes.items()):
                with self.snake_locks[sid]:
                    status = game.update(self.GAME_OVER)
                    if status == 'collision' or status == 'starvation':
                        self.GAME_OVER = True
                        logging.info({"event": "game_over", "reason": status, "snake_id": sid})

            if self.GAME_OVER != True:
                grid, visions, statuses, game_over = self.state()
                logging.info({"event":"frame",
                             "grid": grid, 
                             "visions": visions,
                             "statuses": statuses,
                             "game_over": game_over,
                             "episode": self.episode_number,
                             "frame": self.frame_number})
            elif self.game_over_raised == False:
                snake_lens = {}
                for sid, game in list(self.snakes.items()):
                    with self.snake_locks[sid]:
                        snake_len = len(game.snake)
                    snake_lens[sid] = snake_len

                logging.info({"event": "game_over_results", 
                        "snakes_lengths": snake_lens,
                        "episode": self.episode_number,
                        "frames": self.frame_number
                        })
                self.game_over_raised = True
                    
            if len(self.FOODS) == 0:
                self.spawn_food()
