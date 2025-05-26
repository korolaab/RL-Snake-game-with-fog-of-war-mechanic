import random
import logging

class SnakeGame:
    def __init__(self, snake_id, grid_width, grid_height, foods, snakes, direction=None, vision_radius=5, vision_display_cols=11, vision_display_rows=11):
        self.snake_id = snake_id
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.foods = foods  # set
        self.snakes = snakes  # dict of all snakes
        self.vision_radius = vision_radius
        self.vision_display_cols = vision_display_cols
        self.vision_display_rows = vision_display_rows
        self.direction = direction or (1, 0)
        self.snake = []
        self.ticks = 0
        self.reset()

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
        logging.info(f"Snake {self.snake_id} spawned at {self.snake[0]} dir={self.direction}")

    def relative_turn(self, cmd):
        if cmd == 'left':
            return (self.direction[1], -self.direction[0])
        if cmd == 'right':
            return (-self.direction[1], self.direction[0])
        return self.direction

    def turn(self, cmd):
        self.direction = self.relative_turn(cmd)

    def update(self, game_over):
        if game_over:
            return
        head = self.snake[0]
        new_head = ((head[0] + self.direction[0]) % self.grid_width,
                    (head[1] + self.direction[1]) % self.grid_height)
        occupied = {pos for game in self.snakes.values() for pos in game.snake}
        if new_head in occupied:
            return 'collision'
        self.snake.insert(0, new_head)
        if new_head in self.foods:
            self.foods.remove(new_head)
        else:
            self.snake.pop()
        self.ticks += 1
        return None

    def get_visible_cells(self):
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
                vis[f"{cx},{cy}"] = obj
        return vis

