from utils.seed import set_seed  

from config import parse_args
import config
import os
import sys
from game.manager import GameManager


import time
import struct
import numpy as np
import posix_ipc
import mmap
import json
from collections import defaultdict


def manhattan_cells_without_center(R: int) -> int:
    return 2 * R * (R + 1)




if __name__ == "__main__":
    
    args = parse_args()
    
   
    
    
    # открываем shared memory (создано Clock)
    while True:
        try:
            shm = posix_ipc.SharedMemory("/game_state")
            break
        except posix_ipc.ExistentialError:
            print("[ENV] waiting for /game_state shm...")
            time.sleep(0.1)
    
    # открываем семафоры
    while True:
        try:
            sem_env_tick = posix_ipc.Semaphore("/sem_env_tick")
            break
        except posix_ipc.ExistentialError:
            print("[ENV] waiting for /sem_env semaphore...")
            time.sleep(0.1)
 
    while True:
        try:
            sem_env_done = posix_ipc.Semaphore("/sem_env_done")
            break
        except posix_ipc.ExistentialError:
            print("[ENV] waiting for /sem_env semaphore...")
            time.sleep(0.1)

    regular_dict = json.loads(args.reward_config)
    reward_config = defaultdict(int, regular_dict)  # int() returns 0

    game_manager = GameManager(
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        vision_radius=args.vision_radius,
        vision_display_cols=args.vision_display_cols,
        vision_display_rows=args.vision_display_rows,
        fps=1, 
        seed = args.seed,
        reward_config = reward_config,
        maxStepsWithoutApple = args.max_steps_without_food,
        n_snakes=args.N_snakes
    )

    header_fmt = "d?q"
    header_size = struct.calcsize(header_fmt)
    vision_size = manhattan_cells_without_center(args.vision_radius)
    total_size = header_size + vision_size * 2


    mapfile = mmap.mmap(shm.fd, total_size)
    while True:
        # ждем семафор от Env
        sem_env_tick.acquire()
        print("[ENV] got signal")


        game_manager.step_game_once()

        # пишем данные
        vision = game_manager.snakes[0].getVision()
        reward = game_manager.snakes[0].reward
        struct.pack_into(header_fmt, mapfile,0, reward,False,0)
        mapfile[header_size:header_size+vision.nbytes] = vision.tobytes()
        print("[ENV] wrote state")

        # сигналим Clock, что данные готовы
        sem_env_done.release()

    shm.close_fd()

