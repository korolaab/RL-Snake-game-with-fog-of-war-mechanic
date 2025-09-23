import random
import logging
import threading

class SnakeGame:
    def __init__(self, snake_id, game):
        self.snake_id = snake_id
        self.grid_width = game.GRID_WIDTH
        self.grid_height = game.GRID_HEIGHT
        self.foods = game.FOODS  # set
        self.snakes = game.snakes  # dict of all snakes
        self.vision_radius = game.VISION_RADIUS
        self.vision_display_cols = game.VISION_DISPLAY_COLS
        self.vision_display_rows = game.VISION_DISPLAY_ROWS
        self.direction = (1, 0)
        self.snake = []
        self.ticks = 0
        self.reward_config = game.reward_config
        self.reward = 0
        self.reset()
        self.maxStepsWithoutApple = game.maxStepsWithoutApple
        self.stepsSinceLastApple = 0

        # Simple state management - no internal locking, use external snake_locks
        self._last_known_state = {}

    def find_safe_spawn_location(self):
        occupied = {pos for g in self.snakes.values() for pos in g.snake} | self.foods
        for _ in range(1000):
            head = (random.randrange(self.grid_width), random.randrange(self.grid_height))
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                body = [(head[0] - i*dx, head[1] - i*dy) for i in range(3)]
                if all(0 <= x < self.grid_width and 0 <= y < self.grid_height for x,y in body) and not any(pos in occupied for pos in body):
                    return body, (dx, dy)
        # fallback center spawn
        fallback = [(self.grid_width//2 - i, self.grid_height//2) for i in range(3)]
        return fallback, (1, 0)

    def reset(self):
        spawn_positions, spawn_direction = self.find_safe_spawn_location()
        self.snake = spawn_positions
        self.direction = spawn_direction
        self.ticks = 0
        logging.info({"event": "snake_spawned", "snake_id": self.snake_id, "position": self.snake[0], "direction": self.direction})

    def relative_turn(self, cmd):
        if cmd == 'left':
            return (self.direction[1], -self.direction[0])
        if cmd == 'right':
            return (-self.direction[1], self.direction[0])
        return self.direction

    def turn(self, cmd):
        self.direction = self.relative_turn(cmd)

    def update_vision(self):
        """
        Update visible state - caller must hold snake_locks[sid]
        """
        try:
            self._last_known_state = self._calc_visible_cells()
        except Exception as e:
            logging.error(f"Error updating visible state for snake {self.snake_id}: {e}")
            self._last_known_state = {}

    def get_visible_cells(self):
        """
        Retrieve the last known visible state - caller must hold snake_locks[sid]
        """
        return self._last_known_state.copy()

    def move(self, game_over, ticks=0):
        """
        Handle only movement updates - no vision calculation
        """
        self.reward = 0
        if game_over:
            self.reward += self.reward_config['game_over']
            return ''
        
        head = self.snake[0]
        new_head = ((head[0] + self.direction[0]) % self.grid_width,
                    (head[1] + self.direction[1]) % self.grid_height)
        occupied = {pos for game in self.snakes.values() for pos in game.snake}

        # Legacy collision check: include length=1 condition
        if new_head in occupied or len(self.snake) == 1:
            return 'collision'

        self.snake.insert(0, new_head)
        if new_head in self.foods:
            foods_before = self.foods.copy()
            self.foods.remove(new_head)
            logging.info({"event": "food_consumed", "snake_id": self.snake_id, 
                         "consumed_pos": new_head, "foods_before": list(foods_before), 
                         "foods_after": list(self.foods)})
            self.reward += self.reward_config['eat_food']
            self.stepsSinceLastApple = 0
            # Legacy: no tail removal when eating food (snake grows)
        else:
            # Legacy tick-based tail removal logic
            if ticks == 50:
                self.snake.pop()  # Extra removal every 50 ticks
                logging.debug({"event": "extra_tail_removal", "snake_id": self.snake_id, "ticks": ticks})
            self.snake.pop()  # Always remove tail when no food
            self.stepsSinceLastApple += 1
            logging.debug({"event": "normal_tail_removal", "snake_id": self.snake_id, "snake_length": len(self.snake)})

        self.ticks += 1
        self.reward += self.reward_config['alive']
        return ''

    def update(self, game_over):
        """
        Backward compatibility - calls move() then update_vision()
        """
        status = self.move(game_over)
        if status == '':  # Only update vision if move was successful
            self.update_vision()
        return status

    def _calc_visible_cells(self):
        """
        Calculate visible cells based on snake's current state
        No changes to the original implementation
        """
        head = self.snake[0]
        rotate_map = {
            (0, -1): lambda dx, dy: (dx, dy),
            (1, 0):  lambda dx, dy: (dy, -dx),
            (0, 1):  lambda dx, dy: (-dx, -dy),
            (-1, 0): lambda dx, dy: (-dy, dx)
        }
        rotate = rotate_map.get(self.direction, rotate_map[(0, -1)])
        other_heads = {g.snake[0] for sid, g in self.snakes.items() if sid != self.snake_id and g.snake}
        other_bodies = {pos for sid, g in self.snakes.items() if sid != self.snake_id for pos in g.snake[1:]}
        vis = {}
        for dx in range(-self.vision_radius, self.vision_radius + 1):
            for dy in range(-self.vision_radius, self.vision_radius + 1):
                if abs(dx) + abs(dy) > self.vision_radius:
                    continue
                rx, ry = rotate(dx, dy)
                cx = self.vision_display_cols // 2 + rx
                cy = self.vision_display_rows // 2 + ry
                if not (0 <= cx < self.vision_display_cols and 0 <= cy < self.vision_display_rows):
                    continue
                px = (head[0] + dx) % self.grid_width
                py = (head[1] + dy) % self.grid_height
                pos = (px, py)
                if pos == head:
                    obj = 'HEAD'
                elif pos in self.snake[1:]:
                    obj = 'BODY'
                elif pos in other_heads:
                    obj = 'OTHER_HEAD'
                elif pos in other_bodies:
                    obj = 'OTHER_BODY'
                elif pos in self.foods:
                    obj = 'FOOD'
                else:
                    obj = 'EMPTY'
                flipped_cx = self.vision_display_cols - cx - 1
                vis[f"{flipped_cx},{cy}"] = obj
        return vis
