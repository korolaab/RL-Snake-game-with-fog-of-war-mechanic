from utils.seed import set_seed  

from config import parse_args
import config
import os
import sys
from game import SnakeGame


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
    
   
 
    with open("history.csv",'w') as f:
        print("length", file=f)
    
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
    while True:
        try:
            shm_ctrl = posix_ipc.SharedMemory("/env_control")   
            break
        except posix_ipc.ExistentialError:
            print("[ENV] waiting for /env_control shm...")
            time.sleep(0.1)

    regular_dict = json.loads(args.reward_config)
    reward_config = defaultdict(int, regular_dict)  # int() returns 0

    game = SnakeGame(
        args.grid_width,
        args.grid_height,
        args.vision_radius,
        args.vision_display_cols,
        args.vision_display_rows,
        args.max_lifetime,
        args.max_hunger_steps,
        args.apple_speed
    )

    header_fmt = "<d?q"
    header_size = struct.calcsize(header_fmt)
    vision_size = manhattan_cells_without_center(args.vision_radius) + 2
    total_size = header_size + (vision_size * 8) * 2
    mapfile = mmap.mmap(shm.fd, total_size)

    mapfile_ctrl = mmap.mmap(shm_ctrl.fd, shm_ctrl.size)
    ctrl_fmt = "=iiiii"

    sum_reward = 0
    frames = 0
    while True:
        # ждем семафор от Env
        sem_env_tick.acquire()
        #print("[ENV] got signal")

            # Check if this is a reset request
        (reset,)  = struct.unpack_from("=i", mapfile_ctrl, 0)
        
        if reset == 1:  
            #print("[ENV] Reset requested, resetting environment...")
            #print(f"[ENV] snake_len = {game.max_len}")
            with open("history.csv",'a') as f:
                print(f"{game.max_len}", file=f)

            struct.pack_into(ctrl_fmt, mapfile_ctrl, 0, 0, int(game.max_len), int(game.eaten_apples), int(frames), int(sum_reward))
            game.reset()
            struct.pack_into(header_fmt, mapfile, 0, reward,False,0)
            mapfile[header_size:header_size + state.nbytes] = state.tobytes()
            
            frames = 0
            sum_reward = 0
            
           # print("[ENV] wrote state")
        else:
            reward, game_over, action = struct.unpack_from(header_fmt, mapfile, 0)
            # пишем данные
            sum_reward+=reward
            state, reward, game_over = game.update(action)
            struct.pack_into(header_fmt, mapfile,0, reward,game_over,0)
            mapfile[header_size:header_size + state.nbytes] = state.tobytes()
            frames +=1
            

        # сигналим Clock, что данные готовы
        sem_env_done.release()

    shm.close_fd()

