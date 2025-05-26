"""
Конфигурационный файл для Snake Vision Stream API
"""
import argparse
import numpy as np
import random

# Основные настройки игры (можно менять через аргументы CLI)
GRID_WIDTH = 15
GRID_HEIGHT = 15
VISION_RADIUS = 5
VISION_DISPLAY_COLS = 11
VISION_DISPLAY_ROWS = 11
FPS = 10
MAX_SNAKES = 10

# Парсинг аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description="Snake Vision Stream API configuration")
    parser.add_argument("--grid_width", type=int, default=GRID_WIDTH)
    parser.add_argument("--grid_height", type=int, default=GRID_HEIGHT)
    parser.add_argument("--vision_radius", type=int, default=VISION_RADIUS)
    parser.add_argument("--vision_display_cols", type=int, default=VISION_DISPLAY_COLS)
    parser.add_argument("--vision_display_rows", type=int, default=VISION_DISPLAY_ROWS)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--max_snakes", type=int, default=MAX_SNAKES)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()
