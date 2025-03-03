import pygame
import random
import sys

# --- Настройки игры ---
CELL_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 30
GAME_WIDTH = GRID_WIDTH * CELL_SIZE   # 600 пикселей
GAME_HEIGHT = GRID_HEIGHT * CELL_SIZE   # 600 пикселей
FPS = 10

# --- Настройки окна видения змеи ---
# Вид задаётся по L1-радиусу 5, но отображается со стороны головы.
VISION_RADIUS = 5
# Фиксированная сетка для вида: 11 колонок (от -5 до +5 относительно оси "вперёд") и 6 строк (только вперед: от -5 до 0),
# где голова всегда будет в позиции (5,5).
VISION_DISPLAY_COLS = 11  
VISION_DISPLAY_ROWS = 6   
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
DARKGRAY  = (50, 50, 50)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake + Вид из головы с поворотом")
clock = pygame.time.Clock()

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

def draw_vision_area(snake, food, direction):
    # Для формирования вида из головы поворачиваем координаты так, чтобы "вперёд" всегда было сверху.
    if direction == (0, -1):  # вверх – без поворота
        def rotate(dx, dy): 
            return (dx, dy)
    elif direction == (1, 0):  # вправо – поворот на 90° по часовой стрелке: (dx, dy) -> (dy, -dx)
        def rotate(dx, dy): 
            return (dy, -dx)
    elif direction == (0, 1):  # вниз – поворот на 180°: (dx, dy) -> (-dx, -dy)
        def rotate(dx, dy): 
            return (-dx, -dy)
    elif direction == (-1, 0):  # влево – поворот на 90° против часовой стрелки: (dx, dy) -> (-dy, dx)
        def rotate(dx, dy): 
            return (-dy, dx)
    else:
        def rotate(dx, dy): 
            return (dx, dy)
    
    head_x, head_y = snake[0]
    visible_cells = {}
    
    # Проходим по всем смещениям в пределах L1-радиуса
    for dx in range(-VISION_RADIUS, VISION_RADIUS + 1):
        for dy in range(-VISION_RADIUS, VISION_RADIUS + 1):
            if abs(dx) + abs(dy) > VISION_RADIUS:
                continue
            # Для всех клеток, кроме самой головы, учитываем, что клетка должна быть "перед" головой
            if (dx, dy) != (0, 0):
                if (dx * direction[0] + dy * direction[1]) <= 0:
                    continue
            # Поворачиваем координаты так, чтобы "вперёд" оказалось вверх
            r_x, r_y = rotate(dx, dy)
            # Отображаем только клетки, находящиеся впереди (r_y < 0), кроме головы
            if (dx, dy) != (0, 0) and r_y >= 0:
                continue
            # Фиксируем, что голова (r_x, r_y)=(0,0) будет отображаться в клетке (5,5) видового окна
            disp_col = 5 + r_x
            disp_row = 5 + r_y
            if not (0 <= disp_col < VISION_DISPLAY_COLS and 0 <= disp_row < VISION_DISPLAY_ROWS):
                continue
            
            global_x = (head_x + dx) % GRID_WIDTH
            global_y = (head_y + dy) % GRID_HEIGHT
            cell = (global_x, global_y)
            
            if (dx, dy) == (0, 0):
                color = GREEN
            elif global_x == 0 or global_x == GRID_WIDTH - 1 or global_y == 0 or global_y == GRID_HEIGHT - 1:
                color = BLUE
            elif cell in snake:
                color = DARKGREEN
            elif cell == food:
                color = RED
            else:
                color = WHITE
            
            visible_cells[(disp_col, disp_row)] = color
    
    vision_x_offset = GAME_WIDTH
    vision_y_offset = (WINDOW_HEIGHT - (VISION_DISPLAY_ROWS * VISION_CELL_SIZE)) // 2
    vision_rect = pygame.Rect(vision_x_offset, vision_y_offset, VISION_DISPLAY_COLS * VISION_CELL_SIZE, VISION_DISPLAY_ROWS * VISION_CELL_SIZE)
    pygame.draw.rect(screen, DARKGRAY, vision_rect)
    
    for col in range(VISION_DISPLAY_COLS):
        for row in range(VISION_DISPLAY_ROWS):
            cell_rect = pygame.Rect(vision_x_offset + col * VISION_CELL_SIZE,
                                    vision_y_offset + row * VISION_CELL_SIZE,
                                    VISION_CELL_SIZE, VISION_CELL_SIZE)
            if (col, row) in visible_cells:
                pygame.draw.rect(screen, visible_cells[(col, row)], cell_rect)
            pygame.draw.rect(screen, GRAY, cell_rect, 1)

def main():
    snake = [
        (GRID_WIDTH // 2, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
    ]
    direction = (1, 0)  # начальное направление – вправо
    food = random_food_position(snake)
    game_over = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                # При нажатии клавиш вызываем функцию поворота
                if event.key == pygame.K_UP:
                    direction = turn_snake(direction, "up")
                elif event.key == pygame.K_DOWN:
                    direction = turn_snake(direction, "down")
                elif event.key == pygame.K_LEFT:
                    direction = turn_snake(direction, "left")
                elif event.key == pygame.K_RIGHT:
                    direction = turn_snake(direction, "right")
        
        if not game_over:
            head_x, head_y = snake[0]
            dx, dy = direction
            new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)
            if new_head in snake:
                game_over = True
            else:
                snake.insert(0, new_head)
                if new_head == food:
                    food = random_food_position(snake)
                else:
                    snake.pop()
        
        draw_game_area(snake, food)
        draw_vision_area(snake, food, direction)
        pygame.draw.line(screen, GRAY, (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

