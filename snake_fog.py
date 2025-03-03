import pygame
import random
import sys

# Настройки игры
CELL_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 30
GAME_WIDTH = GRID_WIDTH * CELL_SIZE  # 600 пикселей
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
GRAY      = (100, 100, 100)
DARKGRAY  = (50, 50, 50)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake + Вид змеи")
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
    
    # Отрисовка еды
    food_rect = pygame.Rect(food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, RED, food_rect)
    
    # Отрисовка змейки (голова выделена ярким цветом)
    for i, cell in enumerate(snake):
        x, y = cell
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        color = GREEN if i == 0 else DARKGREEN
        pygame.draw.rect(screen, color, rect)
    
    # Дополнительно: рисуем сетку
    for x in range(0, GAME_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, GAME_HEIGHT))
    for y in range(0, GAME_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (GAME_WIDTH, y))

def draw_vision_area(snake, food):
    # Окно видения (правая часть окна)
    # Чтобы окно видения было не в самом углу, центрируем его по вертикали
    vision_x_offset = GAME_WIDTH  # правая часть начинается после игровой области
    vision_y_offset = (GAME_HEIGHT - VISION_HEIGHT) // 2  # выравнивание по центру по вертикали
    
    # Заполняем область видения фоном для клеток, вне зоны обзора (темно-серым)
    vision_area_rect = pygame.Rect(vision_x_offset, vision_y_offset, VISION_WIDTH, VISION_HEIGHT)
    pygame.draw.rect(screen, DARKGRAY, vision_area_rect)
    
    # Получаем координаты головы змейки
    head_x, head_y = snake[0]
    
    # Проходим по всем клеткам окна видения (11×11)
    for i in range(VISION_GRID_SIZE):
        for j in range(VISION_GRID_SIZE):
            dx = i - VISION_RADIUS  # смещение по горизонтали относительно головы
            dy = j - VISION_RADIUS  # смещение по вертикали относительно головы
            
            # Вычисляем положение клетки в области окна видения
            cell_rect = pygame.Rect(
                vision_x_offset + i * VISION_CELL_SIZE,
                vision_y_offset + j * VISION_CELL_SIZE,
                VISION_CELL_SIZE,
                VISION_CELL_SIZE
            )
            # Если клетка находится в пределах манхэттенского расстояния
            if abs(dx) + abs(dy) <= VISION_RADIUS:
                # Определяем соответствующую клетку в игровой сетке (учитываем обертывание)
                cell_x = (head_x + dx) % GRID_WIDTH
                cell_y = (head_y + dy) % GRID_HEIGHT
                cell = (cell_x, cell_y)
                # Определяем цвет клетки в зависимости от содержимого
                if cell == snake[0]:
                    color = GREEN  # Голова змейки
                elif cell in snake:
                    color = DARKGREEN  # Тело змейки
                elif cell == food:
                    color = RED  # Еда
                else:
                    color = WHITE  # Пустая клетка
            else:
                # Если клетка находится за пределами видимости, оставляем фон DARKGRAY
                color = DARKGRAY
            
            pygame.draw.rect(screen, color, cell_rect)
            # Рисуем рамку клетки
            pygame.draw.rect(screen, GRAY, cell_rect, 1)

def main():
    # Начальное состояние змейки (начинаем с длины 3)
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
            # Проверка столкновения змейки с собой
            if new_head in snake:
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
        
        # Рисуем вертикальную разделительную линию между областями
        pygame.draw.line(screen, GRAY, (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

