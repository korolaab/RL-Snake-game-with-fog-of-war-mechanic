import sys
import time
import random
from collections import deque
from enum import IntEnum, auto

import numpy as np
import pygame


# =========================
# Битовые флаги клетки (uint8)
# =========================
APPLE      = np.uint8(1)   # 0b00000001
SNAKE_BODY = np.uint8(2)   # 0b00000010
SNAKE_HEAD = np.uint8(8)   # 0b00001000

# Для удобства сгруппуем флаги
SNAKE_ANY = np.uint8(SNAKE_BODY | SNAKE_HEAD)


# =========================
# Статус клетки / шага
# =========================
class CellStatus(IntEnum):
    EMPTY = 0
    APPLE = auto()
    CRASH = auto()
    BODY  = auto()
    HEAD  = auto()


# =========================
# Lookup-таблица статусов (0..255)
# =========================
LOOKUP_LIST = [CellStatus.EMPTY] * 256

def _set(cell_value, status):
    LOOKUP_LIST[cell_value] = status

# базовые:
_set(0,              CellStatus.EMPTY)
_set(APPLE,          CellStatus.APPLE)
_set(SNAKE_BODY,     CellStatus.BODY)
_set(SNAKE_HEAD,     CellStatus.HEAD)

# комбинации (если на клетке и змея, и яблоко — трактуем как "съели яблоко",
# но движение в тело/голову — это CRASH)
_set(int(APPLE | SNAKE_BODY), CellStatus.CRASH)  # теоретически мгновенный контакт
_set(int(APPLE | SNAKE_HEAD), CellStatus.CRASH)
_set(int(SNAKE_BODY | SNAKE_HEAD), CellStatus.CRASH)
_set(int(APPLE | SNAKE_BODY | SNAKE_HEAD), CellStatus.CRASH)

LOOKUP_NP = np.array(LOOKUP_LIST, dtype=np.uint8)  # для векторного доступа


# =========================
# Параметры игры
# =========================
W, H      = 10, 10
FPS       = 1  # 2 кадра в секунду
CELL_SIZE = 32
MARGIN    = 1
WIN_W     = W * (CELL_SIZE + MARGIN) + MARGIN
WIN_H     = H * (CELL_SIZE + MARGIN) + MARGIN + 36  # + место под текст

COLOR_BG        = (18, 18, 18)
COLOR_GRID      = (40, 40, 40)
COLOR_APPLE     = (220, 20, 60)
COLOR_BODY      = (80, 200, 120)
COLOR_HEAD      = (255, 215, 0)
COLOR_TEXT      = (220, 220, 220)

RNG = np.random.default_rng()


# =========================
# Вспомогательные функции
# =========================
def wrap(y, x):
    """Тор — телепорт через стены."""
    return y % H, x % W

def spawn_apple(field: np.ndarray):
    """Порождает яблоко на случайной пустой клетке."""
    # пустые клетки: там, где ни одного флага
    empty_mask = field == 0
    ys, xs = np.where(empty_mask)
    if len(ys) == 0:
        return False
    idx = RNG.integers(0, len(ys))
    y, x = int(ys[idx]), int(xs[idx])
    field[y, x] |= APPLE
    return True

def init_game():
    """Создаёт поле, змейку, курс."""
    field = np.zeros((H, W), dtype=np.uint8)

    # стартовая змейка длиной 3 по центру, направо
    cy, cx = H // 2, W // 2
    snake = deque()
    snake.appendleft((cy, cx + 3))  # голова
    snake.append((cy, cx + 2))
    snake.append((cy, cx + 1))
    snake.append((cy, cx + 0))

    # разместим на поле
    for i, (y, x) in enumerate(snake):
        field[y, x] = SNAKE_HEAD if i == 0 else SNAKE_BODY

    # одно яблоко
    spawn_apple(field)

    # курс: вектор (dy, dx). Направо: (0, +1)
    heading = (0, 1)
    return field, snake, heading


# направления относительного поворота
LEFT  = -1
STRAIGHT = 0
RIGHT = 1

def left_of(vec):
    dy, dx = vec
    return (-dx, dy)

def right_of(vec):
    dy, dx = vec
    return (dx, -dy)

def step_direction(heading):
    """Случайно выбрать {влево, прямо, вправо} относительно heading."""
    choice = random.choice((LEFT, STRAIGHT, RIGHT))
    if choice == LEFT:
        return left_of(heading)
    elif choice == RIGHT:
        return right_of(heading)
    else:
        return heading


# =========================
# Проверка клетки (через lookup)
# =========================
def check_cell_value(cell_value: int) -> CellStatus:
    # Используем numpy-lookup как «таблицу переходов».
    return CellStatus(int(LOOKUP_NP[cell_value]))


# =========================
# Игровой шаг
# =========================
def game_step(field: np.ndarray, snake: deque, heading):
    """
    Возвращает:
      ok (bool), ate_apple (bool), new_heading (tuple)
    """
    # 1) Случайно выбрать новое направление (относительно текущего)
    t0 = time.perf_counter()
    new_heading = step_direction(heading)
    t1 = time.perf_counter()

    # 2) Рассчитать новую голову
    hy, hx = snake[0]
    dy, dx = new_heading
    ny, nx = wrap(hy + dy, hx + dx)

    # 3) Проверка клетки назначения
    #    (Важный нюанс — хвост сдвинется, так что заход в «старый хвост» допустим,
    #     если мы НЕ едим яблоко. Для простоты проверим сначала, что в клетке не голова/тело,
    #     а затем при обычном шаге сдвинем хвост.)
    cell_value = int(field[ny, nx])
    # быстрая классификация
    t2 = time.perf_counter()
    status = check_cell_value(cell_value)
    t3 = time.perf_counter()

    ate = False
    ok = True

    if status == CellStatus.HEAD or status == CellStatus.BODY:
        # потенциальная коллизия с собой — но если идём в текущий хвост,
        # и мы НЕ едим яблоко, это допустимо (хвост уйдёт).
        tail_y, tail_x = snake[-1]
        going_into_tail = (ny == tail_y and nx == tail_x)
        if not going_into_tail:
            ok = False
    elif status == CellStatus.CRASH:
        ok = False
    elif status == CellStatus.APPLE:
        ate = True

    # 4) Обновление поля/змейки
    t4 = time.perf_counter()
    if not ok:
        # игра окончена — оставим поле как есть
        update_ms = (time.perf_counter() - t4) * 1000.0
        return False, ate, new_heading, ( (t1-t0)*1000, (t3-t2)*1000, update_ms )

    # ставим новую голову
    snake.appendleft((ny, nx))
    # очистим прежнюю голову -> станет корпусом
    old_head_y, old_head_x = snake[1]
    field[old_head_y, old_head_x] = SNAKE_BODY
    # поставим новую голову
    field[ny, nx] = SNAKE_HEAD

    if ate:
        # хвост не убираем, змейка растёт
        # убираем яблоко битом (на всякий случай)
        field[ny, nx] &= np.uint8(~APPLE)
        # после роста — заспавнить новое яблоко
        spawn_apple(field)
    else:
        # обычный шаг — убираем хвост
        ty, tx = snake.pop()
        field[ty, tx] = np.uint8(0)

    update_ms = (time.perf_counter() - t4) * 1000.0
    return True, ate, new_heading, ( (t1-t0)*1000, (t3-t2)*1000, update_ms )


# =========================
# Отрисовка
# =========================
def draw_field(screen, field, font, timings_ms):
    screen.fill(COLOR_BG)

    # сетка
    for y in range(H):
        for x in range(W):
            v = int(field[y, x])
            rect = pygame.Rect(
                MARGIN + x*(CELL_SIZE+MARGIN),
                36 + MARGIN + y*(CELL_SIZE+MARGIN),
                CELL_SIZE, CELL_SIZE
            )
            # фон клетки
            pygame.draw.rect(screen, COLOR_GRID, rect)

            # содержимое
            if v & APPLE:
                pygame.draw.rect(screen, COLOR_APPLE, rect.inflate(-4, -4))
            if v & SNAKE_BODY:
                pygame.draw.rect(screen, COLOR_BODY, rect.inflate(-6, -6))
            if v & SNAKE_HEAD:
                pygame.draw.rect(screen, COLOR_HEAD, rect.inflate(-10, -10))

    # текст с таймингами
    dir_ms, chk_ms, upd_ms = timings_ms
    text = f"dir: {dir_ms:.2f} ms  check: {chk_ms:.2f} ms  update: {upd_ms:.2f} ms"
    surf = font.render(text, True, COLOR_TEXT)
    screen.blit(surf, (MARGIN, 6))


# =========================
# Main
# =========================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Snake • numpy + bit flags • 2 FPS")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,menlo,monospace", 20)

    field, snake, heading = init_game()

    # стартовые тайминги (для первых кадров)
    timings_ms = (0.0, 0.0, 0.0)

    running = True
    while running:
        # события окна
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # один «шаг» игры
        ok, ate, heading, timings_ms = game_step(field, snake, heading)

        # если проиграли — мгновенно рестарт
        if not ok:
            field, snake, heading = init_game()
            timings_ms = (0.0, 0.0, 0.0)

        # отрисовка
        t_draw0 = time.perf_counter()
        draw_field(screen, field, font, timings_ms)
        pygame.display.flip()
        t_draw1 = time.perf_counter()

        # 2 FPS
        clock.tick(FPS)

        # можно при желании печатать итоговые времена в консоль
        # print(f"draw: {(t_draw1-t_draw0)*1000:.2f} ms")

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

