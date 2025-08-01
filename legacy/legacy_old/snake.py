import pygame
import random
import sys
import math

# Инициализация Pygame
pygame.init()

# Константы игры
CELL_SIZE = 20          # размер клетки (в пикселях)
GRID_WIDTH = 30         # количество клеток по горизонтали
GRID_HEIGHT = 30        # количество клеток по вертикали
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT

FPS = 10                # частота обновления игры (кадров в секунду)
FOG_RADIUS_CELLS = 5    # радиус видимости в клетках
FOG_RADIUS_PIXELS = FOG_RADIUS_CELLS * CELL_SIZE

# Цвета
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
DARKGREEN = (0, 155, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Настройка экрана
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake с ограниченным обзором")
clock = pygame.time.Clock()

# Функция для генерации случайного положения еды
def random_food_position(snake):
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake:
            return pos

# Основной цикл игры
def main():
    # Изначальное положение змейки (начинаем с длины 3)
    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2),
             (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
             (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)]
    direction = (1, 0)  # движение вправо
    food = random_food_position(snake)
    game_over = False

    while True:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Обработка нажатий клавиш для изменения направления
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
            # Обновление положения змейки
            head_x, head_y = snake[0]
            dx, dy = direction
            new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)

            # Проверка столкновения с собой
            if new_head in snake:
                game_over = True
            else:
                snake.insert(0, new_head)
                # Если съели еду, создаём новую еду, иначе удаляем хвост
                if new_head == food:
                    food = random_food_position(snake)
                else:
                    snake.pop()

        # Отрисовка игрового поля
        screen.fill(BLACK)

        # Функция для преобразования координат клетки в пиксели
        def cell_to_pixel(cell):
            x, y = cell
            return x * CELL_SIZE, y * CELL_SIZE

        # Отрисовка еды
        food_pixel = cell_to_pixel(food)
        pygame.draw.rect(screen, RED, (food_pixel[0], food_pixel[1], CELL_SIZE, CELL_SIZE))

        # Отрисовка змейки
        for i, cell in enumerate(snake):
            pixel = cell_to_pixel(cell)
            color = GREEN if i == 0 else DARKGREEN
            pygame.draw.rect(screen, color, (pixel[0], pixel[1], CELL_SIZE, CELL_SIZE))

        # Наложение эффекта "ограниченного обзора"
        # Создаём поверхность для тумана с альфа-каналом
        fog = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), flags=pygame.SRCALPHA)
        fog.fill((0, 0, 0, 220))  # почти непрозрачный чёрный фон

        # Определяем положение центра обзора (голова змейки)
        snake_head_pixel = cell_to_pixel(snake[0])
        # Центрируем круг относительно середины клетки
        center = (snake_head_pixel[0] + CELL_SIZE // 2, snake_head_pixel[1] + CELL_SIZE // 2)

        # Вырезаем круг в тумане, чтобы было видно вокруг головы
        pygame.draw.circle(fog, (0, 0, 0, 0), center, FOG_RADIUS_PIXELS)

        # Накладываем туман на экран
        screen.blit(fog, (0, 0))

        # Отображаем всё на экране
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

