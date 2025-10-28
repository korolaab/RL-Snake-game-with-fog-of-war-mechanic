"""
Configuration settings for the Snake Environment.
"""
import argparse
import numpy as np

GRID_WIDTH = 15
GRID_HEIGHT = 15
VISION_RADIUS = 5
VISION_DISPLAY_COLS = 11
VISION_DISPLAY_ROWS = 11
N_SNAKES = 1
SEED = 1 

def parse_args():
    parser = argparse.ArgumentParser(description="Snake Environment Configurations")
    parser.add_argument("--grid_width", type=int, default=GRID_WIDTH)
    parser.add_argument("--grid_height", type=int, default=GRID_HEIGHT)
    parser.add_argument("--vision_radius", type=int, default=VISION_RADIUS)
    parser.add_argument("--vision_display_cols", type=int, default=VISION_DISPLAY_COLS)
    parser.add_argument("--vision_display_rows", type=int, default=VISION_DISPLAY_ROWS)
    parser.add_argument("--N_snakes", type=int, default=N_SNAKES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument('--reward-config', type=str, default='{"alive": 1}', help='Reward configuration as JSON string. Example: \'{"alive": 1, "food": 10, "death": -10}\'')
    parser.add_argument('--max-steps-without-food', type=int, default=50, help='Maximum steps before game over if no food eaten')
    parser.add_argument('--max-lifetime', type=int, default=10000, help='Maximum steps before snake dies of old age')
    parser.add_argument('--apple-speed', type=float, default=0.5, help='Apple movement speed (0.5 = half speed, moves every 2 frames)')
    return parser.parse_args()
