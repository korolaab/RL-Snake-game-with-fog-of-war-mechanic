# game/manager.py

import threading
import random
import logging
from utils.seed import set_seed  
import os
import sys

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
                       n_snakes=1):
        self.GRID_WIDTH = grid_width
        self.GRID_HEIGHT = grid_height
        self.VISION_RADIUS = vision_radius
        self.VISION_DISPLAY_COLS = vision_display_cols
        self.VISION_DISPLAY_ROWS = vision_display_rows
        self.FPS = fps
        self.MAX_SNAKES = n_snakes
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
        self.ticks = 0          # Legacy: tick counter for tail removal logic
        self.pending_turns = {}  # One turn command per snake per frame
        set_seed(self.seed)
        self.game_over_raised = False
        # Add N_EPISODES from env
        self.max_episodes = int(os.environ.get("N_EPISODES", 10000000))
        # Disable autonomous game loop for synchronous control via /move endpoint
        # Initialize game state but don't start continuous loop
        self.reset_game()

        # threading.Thread(target=self.game_loop, daemon=True).start()

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
        foods_before = self.FOODS.copy()
        occupied = {pos for game in self.snakes.values() for pos in game.snake} | self.FOODS
        while True:
            pos = (random.randint(0, self.GRID_WIDTH - 1), random.randint(0, self.GRID_HEIGHT - 1))
            if pos not in occupied:
                self.FOODS.add(pos)
                logging.info({"event": "food_spawned", "foods_before": list(foods_before), 
                             "new_food_pos": pos, "foods_after": list(self.FOODS), 
                             "frame": self.frame_number})
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
        self.GAME_OVER = False
        self.snakes.clear()
        self.snake_locks.clear()
        self.FOODS.clear()
        self.spawn_food()
        self.game_over_raised = False
        self.episode_number += 1
        self.frame_number = 0
        self.ticks = 0  # Reset ticks counter for new episode
        logging.info({"event": "game_reset", 
                      "action": "all_snakes_removed_food_respawned", 
                      "episode": self.episode_number, 
                      "frame": self.frame_number})
        for i in range(self.MAX_SNAKES):
            self.add_snake(i)
        # Exit logic:
        if self.episode_number >= self.max_episodes:
            logging.info({"event": "env_max_episodes_completed", "total_episodes": self.episode_number})
            sys.exit(0)

    def add_snake(self, snake_id):
        if len(self.snakes) >= self.MAX_SNAKES:
            return False
        #print(f"[ENV] Added snake with id={snake_id}")
        snake = SnakeGame(snake_id, self)
        self.snakes[snake_id] = snake
        self.snake_locks[snake_id] = threading.Lock()
        # Initialize vision for the new snake so it has visible_cells immediately
        with self.snake_locks[snake_id]:
            snake.update_vision()
        return True

    def set_turn_command(self, snake_id, cmd):
        """Set turn command for snake - overwrites if called multiple times per frame"""
        self.pending_turns[snake_id] = cmd

    def remove_snake(self, snake_id):
        if snake_id in self.snakes:
            del self.snakes[snake_id]
        if snake_id in self.snake_locks:
            del self.snake_locks[snake_id]
        if snake_id in self.pending_turns:
            del self.pending_turns[snake_id]

    def get_snake(self, snake_id):
        return self.snakes.get(snake_id)

    def get_lock(self, snake_id):
        return self.snake_locks.get(snake_id)
    
    def update_frame(self):
        """
        Frame-based atomic update: apply turns, then movement, then vision
        """
        # Phase 1: Movement updates (with pending turns)
        for sid, game in list(self.snakes.items()):
            with self.snake_locks[sid]:
                # Apply turn if one was sent this frame
                if sid in self.pending_turns:
                    game.turn(self.pending_turns[sid])
                    
                # Then move with legacy ticks
                status = game.move(self.GAME_OVER, self.ticks)
                if status in ['collision', 'starvation']:
                    self.GAME_OVER = True
                    logging.info({"event": "game_over", "reason": status, "snake_id": sid})
        
        # Phase 2: Vision updates (after all movements)
        for sid, game in self.snakes.items():
            with self.snake_locks[sid]:
                if not self.GAME_OVER:  # Only update vision if game continues
                    game.update_vision()
        
        # Clear all turn commands for next frame
        self.pending_turns.clear()

    def step_game_once(self):
        """Advance game exactly one step - called by /move endpoint for synchronous control"""
        if self.GAME_OVER:
            return
            
        # Increment frame counter (like original)
        self.frame_number += 1
        
        # Legacy: Increment ticks and reset every 50 steps
        self.ticks += 1
        if self.ticks >= 50:
            self.ticks = 0
            logging.debug({"event": "ticks_reset", "frame": self.frame_number})
        
        # Update frame (movement, collision, food consumption, vision)
        self.update_frame()
        
        # Simple food spawning logic - EXACTLY like original game loop
        foods_count = len(self.FOODS)
        logging.info({"event": "checking_food_spawn", "foods_count": foods_count, 
                     "foods_positions": list(self.FOODS), "frame": self.frame_number})
        
        if foods_count == 0:
            logging.info({"event": "triggering_food_spawn", "frame": self.frame_number})
            self.spawn_food()
        
        # Log frame info (moved from game_loop)
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
            # Log game over results (moved from game_loop)
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
            
        logging.debug({"event": "game_stepped", "frame": self.frame_number, "episode": self.episode_number, "foods_count": len(self.FOODS)})


