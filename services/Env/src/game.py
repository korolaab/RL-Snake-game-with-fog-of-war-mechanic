import random
import numpy as np

# --- Colors (RGB) ---
WHITE     = (255, 255, 255)
BLACK     = (0, 0, 0)
GREEN     = (0, 255, 0)    # Snake1 head
DARKGREEN = (0, 155, 0)    # Snake body (own)
BLUE      = (0, 0, 255)    # Snake2 head
DARKBLUE  = (0, 0, 155)    # Other snake body
RED       = (255, 0, 0)
GRAY      = (100, 100, 100)
DARKGRAY  = (50, 51, 50)
ORANGE    = (255, 165, 0)


class RunningApple:
    """Apple that runs away from snake at configurable fraction of snake speed"""

    def __init__(self, position, grid_width, grid_height, speed_fraction=0.5):
        self.position = position
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.move_counter = 0.0
        self.speed_fraction = speed_fraction

    def should_move(self):
        self.move_counter += self.speed_fraction
        if self.move_counter >= 1.0:
            self.move_counter -= 1.0
            return True
        return False

    def get_escape_direction(self, snake_heads, all_body):
        """Calculate best escape direction away from nearest snake head"""
        # Find nearest head
        best_dist = float('inf')
        nearest_head = snake_heads[0]
        for head in snake_heads:
            dx = self.position[0] - head[0]
            dy = self.position[1] - head[1]
            if abs(dx) > self.grid_width / 2:
                dx = -dx / abs(dx) * (self.grid_width - abs(dx)) if dx != 0 else 0
            if abs(dy) > self.grid_height / 2:
                dy = -dy / abs(dy) * (self.grid_height - abs(dy)) if dy != 0 else 0
            dist = abs(dx) + abs(dy)
            if dist < best_dist:
                best_dist = dist
                nearest_head = head

        dx = self.position[0] - nearest_head[0]
        dy = self.position[1] - nearest_head[1]

        if abs(dx) > self.grid_width / 2:
            dx = -dx / abs(dx) * (self.grid_width - abs(dx)) if dx != 0 else 0
        if abs(dy) > self.grid_height / 2:
            dy = -dy / abs(dy) * (self.grid_height - abs(dy)) if dy != 0 else 0

        if abs(dx) > abs(dy):
            move_dir = (1 if dx > 0 else -1, 0)
        elif abs(dy) > abs(dx):
            move_dir = (0, 1 if dy > 0 else -1)
        else:
            move_dir = (1 if dx > 0 else -1, 0)

        new_pos = ((self.position[0] + move_dir[0]) % self.grid_width,
                   (self.position[1] + move_dir[1]) % self.grid_height)

        if new_pos not in all_body:
            return move_dir

        alternatives = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for alt_dir in alternatives:
            alt_pos = ((self.position[0] + alt_dir[0]) % self.grid_width,
                      (self.position[1] + alt_dir[1]) % self.grid_height)
            if alt_pos not in all_body:
                return alt_dir

        return (0, 0)

    def move(self, snake_heads, all_body):
        """Move apple away from nearest snake head"""
        if self.should_move():
            move_dir = self.get_escape_direction(snake_heads, all_body)
            self.position = ((self.position[0] + move_dir[0]) % self.grid_width,
                           (self.position[1] + move_dir[1]) % self.grid_height)

    def respawn(self, all_body):
        """Respawn apple at random location avoiding all snake bodies"""
        while True:
            pos = (random.randint(0, self.grid_width - 1),
                   random.randint(0, self.grid_height - 1))
            if pos not in all_body:
                self.position = pos
                self.move_counter = 0.0
                break

class SnakeGame:
    def __init__(self,
                 GRID_WIDTH,
                 GRID_HEIGHT,
                 VISION_RADIUS,
                 VISION_DISPLAY_COLS,
                 VISION_DISPLAY_ROWS,
                 max_lifetime=10000,
                 max_hunger_steps=150,
                 apple_speed=0.5,
                 num_snakes=1,
                 apple_ttl=0,
                 ):
        self.GRID_WIDTH = GRID_WIDTH
        self.GRID_HEIGHT = GRID_HEIGHT
        self.VISION_RADIUS = VISION_RADIUS
        self.VISION_DISPLAY_COLS = VISION_DISPLAY_COLS
        self.VISION_DISPLAY_ROWS = VISION_DISPLAY_ROWS
        self.num_snakes = num_snakes
        self.ticks = 0

        self.eaten_apples = 0
        self.apple_speed = apple_speed
        self.apple_ttl = apple_ttl
        self.max_hunger_steps = max_hunger_steps
        self.steps_since_food = 0

        self.reset()

    def reset(self):
        # Snake 1: spawns at left-center
        self.snake1 = [
            (self.GRID_WIDTH // 4, self.GRID_HEIGHT // 2),
            (self.GRID_WIDTH // 4 - 1, self.GRID_HEIGHT // 2),
            (self.GRID_WIDTH // 4 - 2, self.GRID_HEIGHT // 2)
        ]
        self.direction1 = (1, 0)
        self.last_action1 = 0

        if self.num_snakes == 2:
            # Snake 2: spawns at right-center, moving left
            self.snake2 = [
                (3 * self.GRID_WIDTH // 4, self.GRID_HEIGHT // 2),
                (3 * self.GRID_WIDTH // 4 + 1, self.GRID_HEIGHT // 2),
                (3 * self.GRID_WIDTH // 4 + 2, self.GRID_HEIGHT // 2)
            ]
            self.direction2 = (-1, 0)
            self.last_action2 = 0

        self.max_len = len(self.snake1)

        all_body = set(self.snake1)
        if self.num_snakes == 2:
            all_body |= set(self.snake2)

        apple_pos = self._random_food_position(all_body)
        self.apple = RunningApple(apple_pos, self.GRID_WIDTH, self.GRID_HEIGHT, self.apple_speed)

        # Two-apple cooperative mode (dual snakes only)
        apple1_pos = apple_pos
        self.apple1 = self.apple
        apple2_pos = self._random_food_position(all_body | {apple1_pos})
        self.apple2 = RunningApple(apple2_pos, self.GRID_WIDTH, self.GRID_HEIGHT, self.apple_speed)

        self.apple1_eaten = False
        self.apple2_eaten = False
        self.apple_ttl_countdown = 0

        self.eaten_apples = 0
        self.steps_since_food = 0

        # For backward compat with single-snake code
        self.snake = self.snake1

    def _random_food_position(self, all_body):
        while True:
            pos = (random.randint(0, self.GRID_WIDTH - 1), random.randint(0, self.GRID_HEIGHT - 1))
            if pos not in all_body:
                return pos

    def random_food_position(self):
        return self._random_food_position(set(self.snake1))

    def _relative_turn(self, direction, turn_command):
        if turn_command == 1:
            return (direction[1], -direction[0])
        elif turn_command == 2:
            return (-direction[1], direction[0])
        else:
            return direction

    def relative_turn(self, turn_command):
        return self._relative_turn(self.direction1, turn_command)

    def _get_all_body(self):
        body = set(self.snake1)
        if self.num_snakes == 2:
            body |= set(self.snake2)
        return body

    def update(self, move1, move2=None):
        if self.num_snakes == 1:
            return self._update_single(move1)
        return self._update_dual(move1, move2)

    def _update_single(self, move):
        self.direction1 = self._relative_turn(self.direction1, move)
        self.last_action1 = move

        head_x, head_y = self.snake1[0]
        dx, dy = self.direction1
        new_head = ((head_x + dx) % self.GRID_WIDTH, (head_y + dy) % self.GRID_HEIGHT)

        if new_head in self.snake1 or len(self.snake1) == 1:
            state = self.get_state()
            return state, 0, True

        self.snake1.insert(0, new_head)
        self.apple.move([self.snake1[0]], set(self.snake1))

        reward = 0.01
        self.steps_since_food += 1
        if new_head == self.apple.position:
            reward += 1
            self.eaten_apples += 1
            self.steps_since_food = 0
            self.apple.respawn(set(self.snake1))
        else:
            self.snake1.pop()

        if self.max_hunger_steps > 0 and self.steps_since_food >= self.max_hunger_steps:
            state = self.get_state()
            return state, 0, True

        self.max_len = max(len(self.snake1), self.max_len)
        self.snake = self.snake1
        state = self.get_state()
        return state, reward, False

    def _update_dual(self, move1, move2):
        self.direction1 = self._relative_turn(self.direction1, move1)
        self.direction2 = self._relative_turn(self.direction2, move2)
        self.last_action1 = move1
        self.last_action2 = move2

        # Move snake1
        h1x, h1y = self.snake1[0]
        dx1, dy1 = self.direction1
        new_head1 = ((h1x + dx1) % self.GRID_WIDTH, (h1y + dy1) % self.GRID_HEIGHT)

        # Move snake2
        h2x, h2y = self.snake2[0]
        dx2, dy2 = self.direction2
        new_head2 = ((h2x + dx2) % self.GRID_WIDTH, (h2y + dy2) % self.GRID_HEIGHT)

        # Check collisions (heads vs all bodies, excluding own head position)
        all_body = set(self.snake1) | set(self.snake2)
        if new_head1 in all_body or new_head2 in all_body or new_head1 == new_head2:
            state1, state2 = self.get_state()
            return (state1, state2), 0, True

        if len(self.snake1) == 1 or len(self.snake2) == 1:
            state1, state2 = self.get_state()
            return (state1, state2), 0, True

        # Add new heads
        self.snake1.insert(0, new_head1)
        self.snake2.insert(0, new_head2)

        all_body_new = set(self.snake1) | set(self.snake2)
        self.apple1.move([new_head1, new_head2], all_body_new)
        self.apple2.move([new_head1, new_head2], all_body_new)

        reward = 0.01
        self.steps_since_food += 1

        # Either snake eats either apple
        if not self.apple1_eaten and (new_head1 == self.apple1.position or new_head2 == self.apple1.position):
            self.apple1_eaten = True
        if not self.apple2_eaten and (new_head1 == self.apple2.position or new_head2 == self.apple2.position):
            self.apple2_eaten = True

        # TTL countdown: starts when first apple eaten
        if (self.apple1_eaten or self.apple2_eaten) and self.apple_ttl > 0:
            if self.apple_ttl_countdown == 0:
                self.apple_ttl_countdown = self.apple_ttl
            else:
                self.apple_ttl_countdown -= 1
                if self.apple_ttl_countdown <= 0:
                    state1, state2 = self.get_state()
                    return (state1, state2), 0, True

        # Both eaten → shared reward + respawn
        if self.apple1_eaten and self.apple2_eaten:
            reward += 1.0
            self.eaten_apples += 1
            self.steps_since_food = 0
            self.apple1_eaten = False
            self.apple2_eaten = False
            self.apple_ttl_countdown = 0
            self.apple1.respawn(all_body_new | {self.apple2.position})
            self.apple2.respawn(all_body_new | {self.apple1.position})
            # Both grow (don't pop)
        else:
            self.snake1.pop()
            self.snake2.pop()

        # Hunger death
        if self.max_hunger_steps > 0 and self.steps_since_food >= self.max_hunger_steps:
            state1, state2 = self.get_state()
            return (state1, state2), 0, True

        self.max_len = max(len(self.snake1), len(self.snake2), self.max_len)
        self.snake = self.snake1
        state1, state2 = self.get_state()
        return (state1, state2), reward, False

    def _get_visible_cells(self, snake, direction, other_snake):
        head_x, head_y = snake[0]
        visible_cells = {}

        if direction == (0, -1):      # Up
            def rotate(dx, dy): return (dx, dy)
        elif direction == (1, 0):     # Right
            def rotate(dx, dy): return (dy, -dx)
        elif direction == (0, 1):     # Down
            def rotate(dx, dy): return (-dx, -dy)
        elif direction == (-1, 0):    # Left
            def rotate(dx, dy): return (-dy, dx)
        else:
            def rotate(dx, dy): return (dx, dy)

        own_body_set = set(snake)
        other_body_set = set(other_snake) if other_snake else set()

        for dx in range(-self.VISION_RADIUS, self.VISION_RADIUS + 1):
            for dy in range(-self.VISION_RADIUS, self.VISION_RADIUS + 1):
                if abs(dx) + abs(dy) > self.VISION_RADIUS:
                    continue

                r_x, r_y = rotate(dx, dy)
                disp_col = (self.VISION_DISPLAY_COLS // 2) + r_x
                disp_row = (self.VISION_DISPLAY_ROWS // 2) + r_y

                if not (0 <= disp_col < self.VISION_DISPLAY_COLS and 0 <= disp_row < self.VISION_DISPLAY_ROWS):
                    continue

                global_x = (head_x + dx) % self.GRID_WIDTH
                global_y = (head_y + dy) % self.GRID_HEIGHT
                cell = (global_x, global_y)

                if (dx, dy) == (0, 0):
                    color = GREEN
                elif cell in own_body_set:
                    color = DARKGREEN
                elif cell in other_body_set:
                    color = DARKBLUE
                elif cell == self.apple1.position:
                    color = RED
                elif cell == self.apple2.position:
                    color = ORANGE
                else:
                    color = WHITE

                visible_cells[(disp_col, disp_row)] = color

        return visible_cells

    def get_visible_cells(self):
        """Backward compat for single snake"""
        return self._get_visible_cells(self.snake1, self.direction1,
                                        self.snake2 if self.num_snakes == 2 else None)

    def _get_state_matrix(self, visible_cells, last_action, snake):
        matrix = []
        for (col, row), color in visible_cells.items():
            if color == WHITE:
                matrix.append([0, 0, 0, 0])
            elif color == DARKGREEN:
                matrix.append([1, 0, 0, 0])
            elif color == DARKBLUE:
                matrix.append([0, 1, 0, 0])
            elif color == RED:
                matrix.append([0, 0, 1, 0])
            elif color == ORANGE:
                matrix.append([0, 0, 0, 1])

        is_alive = np.exp(-np.abs(len(snake)))
        matrix.append([is_alive, 1 - is_alive, 0, 0])

        last_action_vector = [1, 0] if last_action != 1 else [0, 1]
        matrix.append(last_action_vector + [0, 0])

        return np.array(matrix)

    def get_state_matrix(self, visible_cells, last_action):
        return self._get_state_matrix(visible_cells, last_action, self.snake1)

    def get_state(self):
        if self.num_snakes == 1:
            visible_cells = self._get_visible_cells(self.snake1, self.direction1, None)
            state_matrix = self._get_state_matrix(visible_cells, self.last_action1, self.snake1)
            return state_matrix

        # Two snakes: return (state1, state2)
        vis1 = self._get_visible_cells(self.snake1, self.direction1, self.snake2)
        vis2 = self._get_visible_cells(self.snake2, self.direction2, self.snake1)
        state1 = self._get_state_matrix(vis1, self.last_action1, self.snake1)
        state2 = self._get_state_matrix(vis2, self.last_action2, self.snake2)
        return (state1, state2)
