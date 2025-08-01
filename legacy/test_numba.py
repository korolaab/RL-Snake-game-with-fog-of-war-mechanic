import numpy as np
from numba import cuda, float32
import math
import time

# Функция для генерации случайных чисел на GPU
@cuda.jit
def monte_carlo_pi_kernel(rng_states, iterations, out_results):
    """
    Вычисляет значение π методом Монте-Карло на GPU.
    Для каждого потока считаем, сколько случайных точек попало внутрь единичной окружности.
    """
    thread_id = cuda.grid(1)
    
    # Проверяем, не выходит ли поток за границы
    if thread_id < out_results.shape[0]:
        # Счетчик точек внутри единичного круга
        inside_circle = 0
        
        # Генерируем заданное количество случайных точек
        for i in range(iterations):
            # Генерируем случайные координаты от 0 до 1
            x = cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_id)
            y = cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_id)
            
            # Вычисляем расстояние от начала координат
            distance = x*x + y*y
            
            # Если точка находится внутри единичного круга, увеличиваем счетчик
            if distance <= 1.0:
                inside_circle += 1
                
        # Записываем результат для данного потока
        out_results[thread_id] = inside_circle

def calculate_pi_gpu(blocks, threads_per_block, iterations_per_thread):
    """
    Запускает GPU-вычисление значения π.
    
    Args:
        blocks: Количество блоков CUDA
        threads_per_block: Количество потоков на блок
        iterations_per_thread: Количество итераций на каждый поток
        
    Returns:
        float: Приблизительное значение π
    """
    # Общее количество потоков
    threads_total = blocks * threads_per_block
    
    # Создаем массив для результатов каждого потока
    results = np.zeros(threads_total, dtype=np.int32)
    d_results = cuda.to_device(results)
    
    # Инициализируем генератор случайных чисел для GPU
    rng_states = cuda.random.create_xoroshiro128p_states(threads_total, seed=1)
    
    # Замеряем время выполнения
    start_time = time.time()
    
    # Запускаем ядро
    monte_carlo_pi_kernel[blocks, threads_per_block](rng_states, iterations_per_thread, d_results)
    
    # Копируем результаты обратно с GPU
    results = d_results.copy_to_host()
    
    # Сумма всех точек внутри круга
    total_inside = np.sum(results)
    
    # Общее количество сгенерированных точек
    total_points = threads_total * iterations_per_thread
    
    # Вычисляем π (площадь круга составляет π * r²,
    # а площадь квадрата со стороной 2r равна 4r²,
    # отношение точек в круге к общему числу точек ≈ π/4)
    pi_approx = 4.0 * total_inside / total_points
    
    end_time = time.time()
    
    print(f"Время выполнения: {end_time - start_time:.4f} секунд")
    print(f"Всего точек: {total_points:,}")
    print(f"Точек внутри круга: {total_inside:,}")
    print(f"Приблизительное значение π: {pi_approx}")
    print(f"Погрешность: {abs(pi_approx - math.pi):.10f}")
    
    return pi_approx

if __name__ == "__main__":
    # Параметры вычисления
    blocks = 512
    threads_per_block = 256
    iterations_per_thread = 100000
    
    # Запускаем вычисление
    pi_gpu = calculate_pi_gpu(blocks, threads_per_block, iterations_per_thread)
    
    # Выводим точное значение π для сравнения
    print(f"Математическое значение π: {math.pi}")
