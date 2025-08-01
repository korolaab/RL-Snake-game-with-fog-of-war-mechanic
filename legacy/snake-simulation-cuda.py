import numpy as np
from numba import cuda, int32, float32
import math
import random
import time
import matplotlib.pyplot as plt
from matplotlib import colors

# Константы для направлений движения
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Константы для типов ячеек
EMPTY = 0
SNAKE_BODY = 1
SNAKE_HEAD = 2
FOOD = 3
WALL = 4

# Реализация простого генератора случайных чисел для использования на GPU
@cuda.jit(device=True)
def xorshift_rng(state):
    """Простой и быстрый xorshift PRNG для GPU"""
    x = state[0]
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    state[0] = x
    return x

@cuda.jit(device=True)
def random_direction(rng_state):
    """Генерирует случайное направление (0-3)"""
    rand = xorshift_rng(rng_state)
    return abs(rand % 4)

@cuda.jit(device=True)
def is_valid_move(grid, x, y, grid_size):
    """Проверяет, является ли ход допустимым"""
    # Проверка границ
    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        return False
    
    # Проверка столкновения с другими объектами
    cell = grid[y, x]
    return cell == EMPTY or cell == FOOD

@cuda.jit
def update_snakes_kernel(grids, snake_positions, snake_lengths, snake_directions, max_length, 
                         grid_size, rng_states, random_move_prob):
    """
    Ядро CUDA для обновления состояния змеек
    
    Параметры:
    ----------
    grids: 3D массив (n_snakes, grid_size, grid_size)
        Сетки для каждой змейки
    snake_positions: 3D массив (n_snakes, max_length, 2)
        Позиции сегментов каждой змейки (y, x)
    snake_lengths: 1D массив (n_snakes)
        Текущая длина каждой змейки
    snake_directions: 1D массив (n_snakes)
        Текущее направление каждой змейки
    max_length: int
        Максимальная длина змейки
    grid_size: int
        Размер игрового поля
    rng_states: 2D массив (n_snakes, 1)
        Состояния генераторов случайных чисел
    random_move_prob: float
        Вероятность случайного изменения направления (0-1)
    """
    # Получаем индекс змейки
    snake_idx = cuda.grid(1)
    
    # Проверяем, что индекс потока действительный
    if snake_idx >= grids.shape[0]:
        return
    
    # Получаем текущее состояние змейки
    grid = grids[snake_idx]
    positions = snake_positions[snake_idx]
    length = snake_lengths[snake_idx]
    direction = snake_directions[snake_idx]
    rng_state = rng_states[snake_idx]
    
    # Получаем текущую позицию головы
    head_y, head_x = positions[0]
    
    # Решаем, менять ли случайно направление
    rand = xorshift_rng(rng_state) % 100
    if rand < random_move_prob * 100:
        new_direction = random_direction(rng_state)
        # Избегаем разворотов на 180 градусов
        if not (new_direction == UP and direction == DOWN or
                new_direction == DOWN and direction == UP or
                new_direction == LEFT and direction == RIGHT or
                new_direction == RIGHT and direction == LEFT):
            direction = new_direction
    
    # Определяем новую позицию головы на основе направления
    new_head_y, new_head_x = head_y, head_x
    
    if direction == UP:
        new_head_y -= 1
    elif direction == DOWN:
        new_head_y += 1
    elif direction == LEFT:
        new_head_x -= 1
    elif direction == RIGHT:
        new_head_x += 1
    
    # Обрабатываем столкновения со стенами - змейка появляется с другой стороны
    if new_head_x < 0:
        new_head_x = grid_size - 1
    elif new_head_x >= grid_size:
        new_head_x = 0
    
    if new_head_y < 0:
        new_head_y = grid_size - 1
    elif new_head_y >= grid_size:
        new_head_y = 0
    
    # Обрабатываем столкновения с телом змейки
    if grid[new_head_y, new_head_x] == SNAKE_BODY:
        # В случае столкновения с собой, змейка становится короче
        if length > 3:
            # Удаляем последний сегмент
            tail_y, tail_x = positions[length-1]
            grid[tail_y, tail_x] = EMPTY
            snake_lengths[snake_idx] = length - 1
    else:
        # Очищаем сетку от змейки
        for i in range(length):
            y, x = positions[i]
            grid[y, x] = EMPTY
        
        # Сдвигаем все сегменты тела
        for i in range(length-1, 0, -1):
            positions[i][0] = positions[i-1][0]
            positions[i][1] = positions[i-1][1]
        
        # Обновляем позицию головы
        positions[0][0] = new_head_y
        positions[0][1] = new_head_x
        
        # Обновляем направление
        snake_directions[snake_idx] = direction
        
        # Обновляем сетку - добавляем змейку
        for i in range(length):
            y, x = positions[i]
            grid[y, x] = SNAKE_HEAD if i == 0 else SNAKE_BODY

def initialize_simulation(n_snakes, grid_size, initial_length=3):
    """Инициализация симуляции змеек"""
    max_length = grid_size * 2  # Максимальная возможная длина змейки
    
    # Инициализация сеток
    grids = np.zeros((n_snakes, grid_size, grid_size), dtype=np.int32)
    
    # Инициализация позиций змеек
    snake_positions = np.zeros((n_snakes, max_length, 2), dtype=np.int32)
    snake_lengths = np.ones(n_snakes, dtype=np.int32) * initial_length
    snake_directions = np.zeros(n_snakes, dtype=np.int32)
    
    # Инициализация RNG состояний
    rng_states = np.zeros((n_snakes, 1), dtype=np.int32)
    
    # Для каждой змейки
    for i in range(n_snakes):
        # Начальная позиция в случайном месте сетки
        head_x = random.randint(initial_length, grid_size - initial_length)
        head_y = random.randint(initial_length, grid_size - initial_length)
        
        # Случайное начальное направление
        direction = random.randint(0, 3)
        snake_directions[i] = direction
        
        # Инициализация состояния RNG
        rng_states[i, 0] = random.randint(1, 2**31 - 1)
        
        # Создаем начальное тело змейки
        for j in range(initial_length):
            if direction == UP:
                pos_y = head_y + j
                pos_x = head_x
            elif direction == DOWN:
                pos_y = head_y - j
                pos_x = head_x
            elif direction == LEFT:
                pos_y = head_y
                pos_x = head_x + j
            else:  # RIGHT
                pos_y = head_y
                pos_x = head_x - j
            
            # Проверка на выход за границы
            pos_y = pos_y % grid_size
            pos_x = pos_x % grid_size
            
            # Запись позиции
            snake_positions[i, j, 0] = pos_y
            snake_positions[i, j, 1] = pos_x
            
            # Обновление сетки
            grids[i, pos_y, pos_x] = SNAKE_HEAD if j == 0 else SNAKE_BODY
    
    return grids, snake_positions, snake_lengths, snake_directions, rng_states, max_length

def run_simulation(n_snakes, grid_size, n_steps, random_move_prob=0.3, rendering=True):
    """Запуск симуляции змеек на GPU"""
    # Инициализация
    grids, snake_positions, snake_lengths, snake_directions, rng_states, max_length = initialize_simulation(n_snakes, grid_size)
    
    # Копирование данных на устройство
    d_grids = cuda.to_device(grids)
    d_snake_positions = cuda.to_device(snake_positions)
    d_snake_lengths = cuda.to_device(snake_lengths)
    d_snake_directions = cuda.to_device(snake_directions)
    d_rng_states = cuda.to_device(rng_states)
    
    # Настройка запуска CUDA-ядра
    threads_per_block = 128
    blocks = (n_snakes + threads_per_block - 1) // threads_per_block
    
    # Настройка для визуализации
    if rendering:
        # Создаем цветовую карту
        cmap = colors.ListedColormap(['black', 'green', 'red', 'yellow', 'gray'])
        bounds = [EMPTY-0.5, SNAKE_BODY-0.5, SNAKE_HEAD-0.5, FOOD-0.5, WALL-0.5, WALL+0.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Создаем фигуру
        fig, axes = plt.subplots(1, min(4, n_snakes), figsize=(16, 4))
        if n_snakes == 1:
            axes = [axes]
        
        images = []
        for i in range(min(4, n_snakes)):
            img = axes[i].imshow(grids[i], cmap=cmap, norm=norm)
            axes[i].set_title(f'Змейка {i+1}')
            axes[i].axis('off')
            images.append(img)
    
    # Запуск симуляции
    start_time = time.time()
    for step in range(n_steps):
        # Запуск ядра
        update_snakes_kernel[blocks, threads_per_block](d_grids, d_snake_positions, d_snake_lengths, 
                                                        d_snake_directions, max_length, grid_size, 
                                                        d_rng_states, random_move_prob)
        
        # Отображаем результаты каждые 10 шагов
        if rendering and step % 5 == 0:
            # Копируем обновленные сетки обратно на хост
            d_grids.copy_to_host(grids)
            
            # Обновляем визуализацию
            for i in range(min(4, n_snakes)):
                images[i].set_array(grids[i])
            
            plt.suptitle(f'Шаг {step}')
            plt.pause(0.01)
    
    # Копируем результаты обратно на хост
    d_grids.copy_to_host(grids)
    d_snake_lengths.copy_to_host(snake_lengths)
    
    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time:.4f} секунд")
    print(f"Среднее время на шаг: {(end_time - start_time) / n_steps * 1000:.2f} мс")
    
    return grids, snake_lengths

if __name__ == "__main__":
    # Параметры симуляции
    n_snakes = 1000  # Количество змеек
    grid_size = 100   # Размер игрового поля
    n_steps = 500    # Количество шагов симуляции
    random_move_prob = 0.9 # Вероятность случайного изменения направления
    
    
    # Запуск симуляции
    grids, snake_lengths = run_simulation(n_snakes, grid_size, n_steps, random_move_prob, rendering=True)
    print(snake_lengths) 
    # Отображение результатов
    plt.figure(figsize=(10, 5))
    plt.hist(snake_lengths, bins=range(min(snake_lengths), max(snake_lengths)+2), alpha=0.7)
    plt.title('Распределение длин змеек после симуляции')
    plt.xlabel('Длина змейки')
    plt.ylabel('Количество змеек')
    plt.grid(True)
    plt.savefig("plot_cuda.png")
