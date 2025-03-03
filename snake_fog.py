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
# Зона видения задаётся по L1-радиусу 5 (т.е. рассматриваем все клетки (dx,dy) с |dx|+|dy| <= 5)
VISION_RADIUS = 5
# Для отображения вида из головы мы создаём фиксированную сетку:
VISION_DISPLAY_COLS = 11  # горизонтально: от -5 до +5 относительно оси "вперёд"
VISION_DISPLAY_ROWS = 6   # вертикально: от -5 до 0 (только вперед); голова будет в позиции (5,5)
VISION_CELL_SIZE = 20     # размер клетки видения (в пикселях)
VISION_WIDTH = VISION_DISPLAY_COLS * VISION_CELL_SIZE   # 11*20 = 220 пикселей
VISION_HEIGHT = VISION_DISPLAY_ROWS * VISION_CELL_SIZE    # 6*20 = 120 пикселей

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
pygame.display.set_caption("Snake + Вид из головы")
clock = pygame.time.Clock()

# --- Вспомогательная функция для генерации еды ---
def random_food_position(snake):
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake:
            return pos

# --- Отрисовка игровой области (левая часть окна) ---
def draw_game_area(snake, food):
    # Заливаем фон игрового поля
    game_rect = pygame.Rect(0, 0, GAME_WIDTH, GAME_HEIGHT)
    pygame.draw.rect(screen, BLACK, game_rect)
    
    # Рисуем еду
    food_rect = pygame.Rect(food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, RED, food_rect)
    
    # Рисуем змейку (голова выделена ярким зелёным цветом)
    for i, cell in enumerate(snake):
        x, y = cell
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        color = GREEN if i == 0 else DARKGREEN
        pygame.draw.rect(screen, color, rect)
    
    # Рисуем сетку
    for x in range(0, GAME_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, GAME_HEIGHT))
    for y in range(0, GAME_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (GAME_WIDTH, y))

# --- Отрисовка окна видения змеи (правая часть окна) ---
def draw_vision_area(snake, food, direction):
    # Будем отображать только те клетки из множества { (dx,dy) : |dx|+|dy| <= 5 },
    # которые находятся «впереди» головы. Чтобы показать вид из головы, выполняем поворот
    # так, чтобы направление движения змейки оказалось направлено вверх.
    #
    # В итоговой (вращённой) системе координат:
    #   – Голова (0,0) будет отображаться в фиксированной позиции: (5,5) в сетке 11×6.
    #   – Видимыми считаются клетки, у которых r_y < 0 (т.е. они расположены впереди головы),
    #     а сама голова (0,0) всегда видна.
    #
    # Определим функцию поворота (на 90° кратно) в зависимости от направления движения:
    if direction == (0, -1):  # вверх – без поворота
        def rotate(dx, dy): 
            return (dx, dy)
    elif direction == (1, 0):  # вправо – повернуть на 90° по часовой стрелке: (dx,dy) -> (dy, -dx)
        def rotate(dx, dy): 
            return (dy, -dx)
    elif direction == (0, 1):  # вниз – повернуть на 180°: (dx,dy) -> (-dx, -dy)
        def rotate(dx, dy): 
            return (-dx, -dy)
    elif direction == (-1, 0):  # влево – повернуть на 90° против часовой стрелки: (dx,dy) -> (-dy, dx)
        def rotate(dx, dy): 
            return (-dy, dx)
    else:
        def rotate(dx, dy): 
            return (dx, dy)

    head_x, head_y = snake[0]
    # Будем сохранять для каждой отображаемой клетки её цвет в словаре по координатам сетки видения:
    # координаты сетки видения: col от 0 до 10, row от 0 до 5 (где (5,5) – позиция головы).
    visible_cells = {}
    
    # Проходим по всем смещениям (dx, dy) в пределах манхэттенского радиуса 5
    for dx in range(-VISION_RADIUS, VISION_RADIUS + 1):
        for dy in range(-VISION_RADIUS, VISION_RADIUS + 1):
            if abs(dx) + abs(dy) > VISION_RADIUS:
                continue
            # Для всех клеток, кроме самой головы, учитываем условие "видит только вперед"
            if (dx, dy) != (0, 0):
                # Условие по глобальному направлению (dot product > 0)
                if (dx * direction[0] + dy * direction[1]) <= 0:
                    continue
            # Поворачиваем смещение в системе, где вперед – вверх
            r_x, r_y = rotate(dx, dy)
            # В системе взгляда клетки должны находиться впереди (r_y < 0), кроме головы
            if (dx, dy) != (0, 0) and r_y >= 0:
                continue
            # Определяем позицию в сетке видения.
            # Фиксируем, что голова (r = (0,0)) будет в клетке (5,5)
            disp_col = 5 + r_x
            disp_row = 5 + r_y
            # Отображать будем только если попадаем в пределы заданной сетки (11 столбцов, 6 строк: row 0..5)
            if not (0 <= disp_col < VISION_DISPLAY_COLS and 0 <= disp_row < VISION_DISPLAY_ROWS):
                continue
            
            # Глобальные координаты соответствующей клетки (учитывая обёртывание поля)
            global_x = (head_x + dx) % GRID_WIDTH
            global_y = (head_y + dy) % GRID_HEIGHT
            cell = (global_x, global_y)
            
            # Определяем цвет клетки:
            if (dx, dy) == (0, 0):
                color = GREEN  # голова
            elif global_x == 0 or global_x == GRID_WIDTH - 1 or global_y == 0 or global_y == GRID_HEIGHT - 1:
                color = BLUE   # граница поля
            elif cell in snake:
                color = DARKGREEN
            elif cell == food:
                color = RED
            else:
                color = WHITE
            
            visible_cells[(disp_col, disp_row)] = color

    # --- Отрисовка видовой области ---
    # Выравниваем область видения по вертикали: правый блок начинается сразу после игровой области
    vision_x_offset = GAME_WIDTH
    # Центрируем область видения по высоте окна (высота области = VISION_DISPLAY_ROWS * VISION_CELL_SIZE)
    vision_y_offset = (WINDOW_HEIGHT - (VISION_DISPLAY_ROWS * VISION_CELL_SIZE)) // 2

    # Заливаем фон области видения (все клетки, по умолчанию, DARKGRAY)
    vision_rect = pygame.Rect(vision_x_offset, vision_y_offset, VISION_DISPLAY_COLS * VISION_CELL_SIZE, VISION_DISPLAY_ROWS * VISION_CELL_SIZE)
    pygame.draw.rect(screen, DARKGRAY, vision_rect)

    # Проходим по каждой клетке фиксированной сетки видения и отрисовываем её
    for col in range(VISION_DISPLAY_COLS):
        for row in range(VISION_DISPLAY_ROWS):
            cell_rect = pygame.Rect(vision_x_offset + col * VISION_CELL_SIZE,
                                    vision_y_offset + row * VISION_CELL_SIZE,
                                    VISION_CELL_SIZE, VISION_CELL_SIZE)
            if (col, row) in visible_cells:
                pygame.draw.rect(screen, visible_cells[(col, row)], cell_rect)
            pygame.draw.rect(screen, GRAY, cell_rect, 1)

# --- Основной цикл игры ---
def main():
    # Начальное состояние змейки (начинаем с длины 3)
    snake = [
        (GRID_WIDTH // 2, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
    ]
    direction = (1, 0)  # стартовое направление – вправо
    food = random_food_position(snake)
    game_over = False

    while True:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                # Изменение направления (змея не знает своего азимута, поэтому вид в правой части поворачивать не будем)
                if event.key == pygame.K_UP and direction != (0, 1):
                    direction = (0, -1)
                elif event.key == pygame.K_DOWN and direction != (0, -1):
                    direction = (0, 1)
                elif event.key == pygame.K_LEFT and direction != (1, 0):
                    direction = (-1, 0)
                elif event.key == pygame.K_RIGHT and direction != (-1, 0):
                    direction = (1, 0)
        
        if not game_over:
            head_x, head_y = snake[0]
            dx, dy = direction
            new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)
            # Если змея столкнулась с собой – конец игры
            if new_head in snake:
                game_over = True
            else:
                snake.insert(0, new_head)
                if new_head == food:
                    food = random_food_position(snake)
                else:
                    snake.pop()
        
        # Отрисовка игровой области (левая часть)
        draw_game_area(snake, food)
        # Отрисовка вида из головы змеи (правая часть) – с поворотом, чтобы вперед всегда было сверху
        draw_vision_area(snake, food, direction)
        
        # Рисуем вертикальную разделительную линию между областями
        pygame.draw.line(screen, GRAY, (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

