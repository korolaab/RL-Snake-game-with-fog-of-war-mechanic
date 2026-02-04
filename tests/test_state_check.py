import numpy as np
import timeit
from enum import IntEnum

# === Константы для битов ===
APPLE = 1   # 0b0001
SNAKE = 2   # 0b0010
WALL  = 4   # 0b0100


# === Enum для результата ===
class CellStatus(IntEnum):
    EMPTY = 0
    APPLE = 1
    CRASH = 2


# === Способ 1: if ===
def check_if(cell: int) -> CellStatus:
    if cell & APPLE:
        return CellStatus.APPLE
    if cell & (SNAKE | WALL):
        return CellStatus.CRASH
    return CellStatus.EMPTY


# === Способ 2: dict lookup ===
STATUS_MAP = {
    APPLE: CellStatus.APPLE,
    SNAKE: CellStatus.CRASH,
    WALL:  CellStatus.CRASH,
    APPLE | SNAKE: CellStatus.CRASH,
    APPLE | WALL: CellStatus.CRASH,
    SNAKE | WALL: CellStatus.CRASH,
    APPLE | SNAKE | WALL: CellStatus.CRASH,
}

def check_dict(cell: int) -> CellStatus:
    return STATUS_MAP.get(cell, CellStatus.EMPTY)


# === Способ 3: list lookup (0..255) ===
LOOKUP_LIST = [CellStatus.EMPTY] * 256
LOOKUP_LIST[APPLE] = CellStatus.APPLE
LOOKUP_LIST[SNAKE] = CellStatus.CRASH
LOOKUP_LIST[WALL]  = CellStatus.CRASH
LOOKUP_LIST[APPLE | SNAKE] = CellStatus.CRASH
LOOKUP_LIST[APPLE | WALL] = CellStatus.CRASH
LOOKUP_LIST[SNAKE | WALL] = CellStatus.CRASH
LOOKUP_LIST[APPLE | SNAKE | WALL] = CellStatus.CRASH

def check_list(cell: int) -> CellStatus:
    return LOOKUP_LIST[cell]


# === Способ 4: numpy mask + векторизация ===
LOOKUP_NP = np.array(LOOKUP_LIST, dtype=np.uint8)

def check_numpy(field: np.ndarray) -> np.ndarray:
    """Векторная проверка всего массива"""
    return LOOKUP_NP[field]


# === Тестовое поле ===
N = 1000
field = np.random.choice([0, APPLE, SNAKE, WALL,
                          APPLE | SNAKE, APPLE | WALL],
                         size=(N, N)).astype(np.uint8)

# Одно значение для «if/dict/list»
test_cells = field.ravel().tolist()


# === Бенчмарк ===
def bench_if():
    for c in test_cells:
        check_if(c)

def bench_dict():
    for c in test_cells:
        check_dict(c)

def bench_list():
    for c in test_cells:
        check_list(c)

def bench_numpy():
    check_numpy(field)


if __name__ == "__main__":
    for name, fn in [
        ("if", bench_if),
        ("dict", bench_dict),
        ("list", bench_list),
        ("numpy", bench_numpy),
    ]:
        t = timeit.timeit(fn, number=10)
        print(f"{name:6s}: {t:.4f} sec")

