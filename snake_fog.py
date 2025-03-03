import pygame
import random
import sys

# Настройки игры
CELL_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 30
GAME_WIDTH = GRID_WIDTH * CELL_SIZE   # 600 пикселей
GAME_HEIGHT = GRID_HEIGHT * CELL_SIZE  # 600 пикселей

FPS = 10

# Настройки видения змеи (манхэттенский радиус 5)
VISION_RADIUS = 5
VISION_GRID_SIZE = 2 * VISION_RADIUS + 1  # 11 клеток
VISION_CELL_SIZE = 20  # размер клетки в окне видения
VISION_WIDTH = VISION_GRID_SIZE * VISION_CELL_SIZE  # 220 пикселей
VISION_HEIGHT = VISION_GRID_SIZE * VISION_CELL_SIZE  # 220 пикселей

# Размеры общего окна (игровая область слева + окно видения справа)
WINDOW_WIDTH = GAME_WIDTH + VISION_WIDTH  # 600 + 220 = 820 пикселей
WINDOW_HEIGHT = GAME_HEIGHT  # 600 пикселей

# Цвета (RGB)
WHITE     = (255, 255, 255)
BLACK     = (0, 0, 0)
GREEN     = (0, 255, 0)
DARKGREEN = (0, 155, 0)
RED       = (255, 0, 0)
BLUE      = (0, 0, 255)      # Для отображения столкновения с границей
GRAY      = (100, 100, 100)
DARKGRAY  = (50, 50, 50)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake + Вид змеи (граница = смерть)")
clock = pygame.time.Clock()

def random_food_position(snake):
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake:
            return pos

def draw_game_area(snake, food):
    # Игровая область (левая часть окна)
    game_rect = pygame.Rect(0, 0, GAME_WIDTH, GAME_HEIGHT)
    pygame.draw.rect(screen, BLACK, game_rect)
    
    # Отрисовка еды (если еда находится в пределах поля)
    if 0 <= food[0] < GRID_WIDTH and 0 <= food[1] < GRID_HEIGHT:
        food_rect = pygame.Rect(food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, RED, food_rect)
    
    # Отрисовка змейки
    for i, cell in enumerate(snake):
        x, y = cell
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        color = GREEN if i == 0 else DARKGREEN
        pygame.draw.rect(screen, color, rect)
    
    # Рисуем сетку для удобства
    for x in range(0, GAME_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, GAME_HEIGHT))
    for y in range(0, GAME_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (GAME_WIDTH, y))

def draw_vision_area(snake, food):
    # Окно видения (правая часть окна)
    vision_x_offset = GAME_WIDTH  # начало правой части
    vision_y_offset = (GAME_HEIGHT - VISION_HEIGHT) // 2  # выравнивание по центру по вертикали
    
    # Заполняем область видения базовым фоном (темно-серым)
    vision_area_rect = pygame.Rect(vision_x_offset, vision_y_offset, VISION_WIDTH, VISION_HEIGHT)
    pygame.draw.rect(screen, DARKGRAY, vision_area_rect)
    
    # Получаем координаты головы змейки
    head_x, head_y = snake[0]
    
    # Отрисовываем сетку видения (11x11 клеток)
    for i in range(VISION_GRID_SIZE):
        for j in range(VISION_GRID_SIZE):
            dx = i - VISION_RADIUS  # смещение относительно головы
            dy = j - VISION_RADIUS
            cell_rect = pygame.Rect(
                vision_x_offset + i * VISION_CELL_SIZE,
                vision_y_offset + j * VISION_CELL_SIZE,
                VISION_CELL_SIZE,
                VISION_CELL_SIZE
            )
            # Если клетка находится в пределах манхэттенского расстояния
            if abs(dx) + abs(dy) <= VISION_RADIUS:
                nx = head_x + dx
                ny = head_y + dy
                # Если клетка выходит за пределы игрового поля – окрашиваем в синий
                if nx < 0 or nx >= GRID_WIDTH or ny < 0 or ny >= GRID_HEIGHT:
                    color = BLUE
                else:
                    cell = (nx, ny)
                    if cell == snake[0]:
                        color = GREEN   # Голова змейки
                    elif cell in snake:
                        color = DARKGREEN  # Тело змейки
                    elif cell == food:
                        color = RED    # Еда
                    else:
                        color = WHITE  # Пустая клетка
            else:
                # За пределами видимости
                color = DARKGRAY
            pygame.draw.rect(screen, color, cell_rect)
            pygame.draw.rect(screen, GRAY, cell_rect, 1)

def main():
    # Начальное состояние змейки (длина 3)
    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2),
             (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
             (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)]
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
                # Игровое управление не работает после game_over
                if not game_over:
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
            new_head = (head_x + dx, head_y + dy)
            # Проверка столкновения с границей
            if new_head[0] < 0 or new_head[0] >= GRID_WIDTH or new_head[1] < 0 or new_head[1] >= GRID_HEIGHT:
                snake.insert(0, new_head)
                game_over = True
            # Проверка столкновения с телом змейки
            elif new_head in snake:
                game_over = True
            else:
                snake.insert(0, new_head)
                if new_head == food:
                    food = random_food_position(snake)
                else:
                    snake.pop()
        
        # Отрисовка левой части – игровая область
        draw_game_area(snake, food)
        # Отрисовка правой части – окно видения змеи
        draw_vision_area(snake, food)
        
        # Разделительная вертикальная линия между областями
        pygame.draw.line(screen, GRAY, (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

