import pygame
import random
import sys

# --- Настройки игры ---
CELL_SIZE = 20
GRID_WIDTH = 15
GRID_HEIGHT = 15
GAME_WIDTH = GRID_WIDTH * CELL_SIZE   # 600 пикселей
GAME_HEIGHT = GRID_HEIGHT * CELL_SIZE   # 600 пикселей
FPS = 1

# --- Настройки окна видения змеи ---
# Вид задаётся по L1-радиусу 5, но отображается со стороны головы.
VISION_RADIUS = 5
# Фиксированная сетка для вида: 11 колонок (от -5 до +5 относительно оси "вперёд") и 6 строк (только вперед: от -5 до 0),
# где голова всегда будет отображаться в клетке (5,5)
VISION_DISPLAY_COLS = 11 
VISION_DISPLAY_ROWS = 11   
VISION_CELL_SIZE = 20     
VISION_WIDTH = VISION_DISPLAY_COLS * VISION_CELL_SIZE   # 220 пикселей
VISION_HEIGHT = VISION_DISPLAY_ROWS * VISION_CELL_SIZE    # 120 пикселей

# --- Общие размеры окна ---
WINDOW_WIDTH = GAME_WIDTH + VISION_WIDTH   # 600 + 220 = 820 пикселей
WINDOW_HEIGHT = GAME_HEIGHT                # 600 пикселей

# --- Цвета (RGB) ---
WHITE     = (255, 255, 255)
BLACK     = (0, 0, 0)
GREEN     = (0, 255, 0)    # Голова змейки
DARKGREEN = (0, 155, 0)    # Тело змейки
RED       = (255, 0, 0)
BLUE      = (0, 0, 255)    # Граница поля
GRAY      = (100, 100, 100)
DARKGRAY  = (50, 51, 50)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake + Вид из головы с поворотом")
clock = pygame.time.Clock()

# --- Шрифт для отображения очков ---
score_font = pygame.font.SysFont("Arial", 24)

def random_food_position(snake):
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake:
            return pos

# --- Функция для поворота змейки ---
def turn_snake(current_direction, command):
    """
    Поворачивает змейку в заданное абсолютное направление,
    если новое направление не является обратным текущему движению.
    
    current_direction: кортеж (dx, dy) – текущее направление.
    command: строка "up", "down", "left" или "right".
    """
    mapping = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0)
    }
    new_direction = mapping.get(command, current_direction)
    # Если новое направление является обратным текущему, возвращаем текущее направление.
    if (current_direction[0] + new_direction[0] == 0 and 
        current_direction[1] + new_direction[1] == 0) and current_direction != (0, 0):
        return current_direction
    return new_direction

def draw_game_area(snake, food):
    game_rect = pygame.Rect(0, 0, GAME_WIDTH, GAME_HEIGHT)
    pygame.draw.rect(screen, BLACK, game_rect)
    
    food_rect = pygame.Rect(food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, RED, food_rect)
    
    for i, cell in enumerate(snake):
        x, y = cell
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        color = GREEN if i == 0 else DARKGREEN
        pygame.draw.rect(screen, color, rect)
    
    for x in range(0, GAME_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, GAME_HEIGHT))
    for y in range(0, GAME_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (GAME_WIDTH, y))

visible_state = ""
def draw_vision_area(snake, food, direction):
    # Функция поворота координат для ориентации взгляда змейки:
    # Поворот осуществляется так, чтобы независимо от текущего направления змейки,
    # "вперёд" всегда располагалось вверх в видовой области.
    if direction == (0, -1):  # вверх – без поворота
        def rotate(dx, dy): 
            return (dx, dy)
    elif direction == (1, 0):  # вправо – поворот на 90° по часовой стрелке: (dx,dy) -> (dy, -dx)
        def rotate(dx, dy): 
            return (dy, -dx)
    elif direction == (0, 1):  # вниз – поворот на 180°: (dx,dy) -> (-dx, -dy)
        def rotate(dx, dy): 
            return (-dx, -dy)
    elif direction == (-1, 0):  # влево – поворот на 90° против часовой стрелки: (dx,dy) -> (-dy, dx)
        def rotate(dx, dy): 
            return (-dy, dx)
    else:
        def rotate(dx, dy):
            return (dx, dy)
    
    head_x, head_y = snake[0]
    visible_cells = {}
    global visible_state
    visible_state = ""
    # Перебираем все клетки, удовлетворяющие |dx|+|dy| ≤ VISION_RADIUS.
    for dx in range(-VISION_RADIUS, VISION_RADIUS + 1):
        for dy in range(-VISION_RADIUS, VISION_RADIUS + 1):
            if abs(dx) + abs(dy) > VISION_RADIUS:
                continue
            # Здесь больше не отсекаем "непередовые" клетки – показываем весь ромб.
            r_x, r_y = rotate(dx, dy)
            # Фиксируем, что голова (0,0) всегда будет отображаться в центре видового окна.
            disp_col = (VISION_DISPLAY_COLS // 2) + r_x
            disp_row = (VISION_DISPLAY_ROWS // 2) + r_y
            if not (0 <= disp_col < VISION_DISPLAY_COLS and 0 <= disp_row < VISION_DISPLAY_ROWS):
                continue
            
            global_x = (head_x + dx) % GRID_WIDTH
            global_y = (head_y + dy) % GRID_HEIGHT
            cell = (global_x, global_y)
            
            # Определяем цвет клетки:
            if (dx, dy) == (0, 0):
                color = GREEN  # голова
                c = 'H'
            #elif global_x == 0 or global_x == GRID_WIDTH - 1 or global_y == 0 or global_y == GRID_HEIGHT - 1:
            #    color = WHITE   # граница поля
            elif cell in snake:
                color = DARKGREEN
                c = 'S'
            elif cell == food:
                color = RED
                c = 'A'
            else:
                color = WHITE
                c = '_'
            
            visible_cells[(disp_col, disp_row)] = color
            visible_state += c
    # Вычисляем смещение для области видения (она расположена справа от игровой области)
    vision_x_offset = GAME_WIDTH
    vision_y_offset = (WINDOW_HEIGHT - (VISION_DISPLAY_ROWS * VISION_CELL_SIZE)) // 2
    vision_rect = pygame.Rect(vision_x_offset, vision_y_offset, VISION_DISPLAY_COLS * VISION_CELL_SIZE, VISION_DISPLAY_ROWS * VISION_CELL_SIZE)
    pygame.draw.rect(screen, DARKGRAY, vision_rect)
    
    # Отрисовка клеток видовой области
    for col in range(VISION_DISPLAY_COLS):
        for row in range(VISION_DISPLAY_ROWS):
            cell_rect = pygame.Rect(vision_x_offset + col * VISION_CELL_SIZE,
                                    vision_y_offset + row * VISION_CELL_SIZE,
                                    VISION_CELL_SIZE, VISION_CELL_SIZE)
            if (col, row) in visible_cells:
                pygame.draw.rect(screen, visible_cells[(col, row)], cell_rect)
            pygame.draw.rect(screen, GRAY, cell_rect, 1)


from collections import defaultdict
import numpy as np
from tabulate import tabulate

class QLearner():
    def __init__(self, epsilon=0.2, learning_rate = 1, gamma = 0.999):
        self.epsilon = epsilon
        self.q_values = defaultdict(lambda: np.zeros(3, np.float64))
        self.last_state = ''
        self.last_action = 0
        self.gamma = gamma
        self.learning_rate = learning_rate

    def _egreedy_policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(3) 
        else:
            return np.argmax(self.q_values[state])
    
    def action(self,state):
        self.last_action = self._egreedy_policy(state)
        return self.last_action

    def update(self, next_state, reward, done): 
        td_target = reward + self.gamma * max(self.q_values[next_state]) 
        td_error =  td_target - self.q_values[self.last_state][self.last_action]
        self.q_values[self.last_state][self.last_action] += self.learning_rate * td_error
        if(done):
            print(self.last_state)
            self.q_values[self.last_state][self.last_action] = reward
            print(self.q_values[self.last_state])
        self.last_state = next_state

    def q_values_table_str(self):
        # Convert dictionary to list of tuples
        table = []
        for key, arr in self.q_values.items():
            table.append([key] + list(arr))
        return table

import csv

def relative_turn(direction, turn_command):
    """
    Принимает текущий вектор направления (dx, dy) и строку "right" или "left",
    возвращает новый вектор направления, повернутый на 90° относительно текущего.
    
    В нашей системе координат (y увеличивается вниз):
      - Поворот "left" (от лица змейки) соответствует: (dx, dy) -> (dy, -dx)
      - Поворот "right" соответствует: (dx, dy) -> (-dy, dx)
    Если turn_command не равен "left" или "right", возвращается исходное направление.
    """
    if turn_command == "left":
        return (direction[1], -direction[0])
    elif turn_command == "right":
        return (-direction[1], direction[0])
    else:
        return direction

def main():
    # Инициализация начального состояния
    #random.seed(10)
    avg_score = 0  
    N_iterations = 1
    snake = [
        (GRID_WIDTH // 2, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
    ]
    direction = (1, 0)  # начальное направление – вправо
    food = random_food_position(snake)
    game_over = False
    game_over_time = None
    global visible_state
    qlearner = QLearner(epsilon = 0.2, learning_rate=1, gamma=0.9)

    
    # Define a custom event triggered every 1 second
    TIMER_QSTATE_SAVE_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(TIMER_QSTATE_SAVE_EVENT, 5000)  # 1000ms = 1 second

    last_score = 0
    while True:
        # Очищаем весь экран в начале каждого кадра, чтобы не было "накладывания" старых рисунков
        screen.fill(BLACK)
        #print(visible_state) 
        go = ["straight","left", "right"]
        if game_over == False:
            action = qlearner.action(visible_state)
#            direction = relative_turn(direction, go[action]) 
        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == TIMER_QSTATE_SAVE_EVENT:
                with open("q_values.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["state","straight", "left", "right"])  # Header
                    writer.writerows(qlearner.q_values_table_str())
                print("saved_q_state")
            elif event.type == pygame.KEYDOWN:
                # Изменяем направление только если игра не окончена
                if not game_over:
                   # if event.key == pygame.K_UP:
                   #     direction = turn_snake(direction, "up")
                   # elif event.key == pygame.K_DOWN:
                   #     direction = turn_snake(direction, "down")
                    if event.key == pygame.K_LEFT:
                        direction = relative_turn(direction, "left")
                    elif event.key == pygame.K_RIGHT:
                        direction = relative_turn(direction, "right")
        
        if not game_over:
            head_x, head_y = snake[0]
            dx, dy = direction
            new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)
            if new_head in snake:
                #random.seed(10)
                print(f"reward: {len(snake) - 3}")
                #print(visible_state)
                qlearner.update(visible_state, -(len(snake) -3),True)
                game_over = True
                N_iterations += 1
                avg_score = (len(snake) - 3)
                game_over_time = pygame.time.get_ticks()
                
            else:
                snake.insert(0, new_head)
                if new_head == food:
                    food = random_food_position(snake)
                else:
                    snake.pop()
        else:
            # Автопрезапуск игры через 0.5 секунды после game over
            if pygame.time.get_ticks() - game_over_time >= 100:
                snake = [
                    (GRID_WIDTH // 2, GRID_HEIGHT // 2),
                    (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
                    (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
                ]
                direction = (1, 0)
                food = random_food_position(snake)
                game_over = False
                last_score = 0
        
        #draw_game_area(snake, food)
        draw_vision_area(snake, food, direction)
        pygame.draw.line(screen, GRAY, (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
        
        # Выводим текущее количество очков в правом верхнем углу
        score = len(snake) - 3  # очки = съеденная еда
        if game_over == False :
            qlearner.update(visible_state, score - last_score, False)
        last_score = score

        score_text = score_font.render(f"avg: {avg_score/N_iterations:.2f} score: {score}", True, WHITE)
        score_rect = score_text.get_rect()
        score_rect.topright = (WINDOW_WIDTH - 10, 10)
        screen.blit(score_text, score_rect)
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

