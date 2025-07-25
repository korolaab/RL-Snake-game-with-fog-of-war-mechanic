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

# TCP Synchronization defaults
SYNC_ENABLED = False
SYNC_PORT = 5555
SYNC_BUFFER_SIZE = 1024

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
    parser.add_argument('--reward-config', type=str, default='{"alive": 1}', help='Reward configuration as JSON string')
    parser.add_argument('--max-steps-without-food', type=int, default=50, help='Maximum steps before game over if no food eaten')
    
    # TCP Synchronization arguments
    parser.add_argument('--sync_enabled', action='store_true',
                        help='Enable TCP synchronization for game loop timing')
    parser.add_argument('--sync_host', type=str, default='sync_service_host',
                        help='Host to connect for TCP synchronization signals')
    parser.add_argument('--sync_port', type=int, default=5555,
                        help='Port to connect for TCP synchronization signals')
    parser.add_argument('--sync_buffer_size', type=int, default=1024,
                        help='Buffer size for TCP socket')
    
    return parser.parse_args()