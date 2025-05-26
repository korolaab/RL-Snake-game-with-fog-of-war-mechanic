# game/manager.py

import threading
import random
import logging

from .snake import SnakeGame

class GameManager:
    def __init__(self, grid_width, grid_height, vision_radius, vision_display_cols, vision_display_rows, fps, max_snakes=10):
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
        # Дать стримам завершиться
        import time
        time.sleep(0.1)
        with self.game_over_lock:
            self.GAME_OVER = False
        self.snakes.clear()
        self.snake_locks.clear()
        self.FOODS.clear()
        self.spawn_food()
        logging.info("Game reset: all snakes removed, food respawned.")

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
        while True:
            time.sleep(1.0 / self.FPS)
            for sid, game in list(self.snakes.items()):
                with self.snake_locks[sid]:
                    game.update()

