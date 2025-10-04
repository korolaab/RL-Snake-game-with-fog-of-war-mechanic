import requests
import time
import json
import argparse
import sys
import os
# from snake_agent import SnakeAgent
from datetime import datetime
import logging

import posix_ipc
import mmap
import struct



from collections import defaultdict
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural network agent local training only (no gRPC)")
    parser.add_argument("--snake-id", type=int, required=True, help="Snake ID for this agent")
    parser.add_argument("--log-file", type=str, default="agent_log.json", help="Log file path")
    parser.add_argument("--env-host", type=str, default="localhost:5000", help="Environment host URL")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=5, help="Episodes per batch")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (gamma) for RL")
    parser.add_argument("--beta", type=float, default=0.1, help="Entropy bonus (beta)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Number of episodes before exit (overrides env N_EPISODES)")

    args = parser.parse_args()

    

    
    # neural_agent_local(
    #         snake_id=args.snake_id,
    #         log_file=args.log_file,
    #         env_host=args.env_host,
    #         model_save_dir=args.model_dir,
    #         learning_rate=args.learning_rate,
    #         batch_size=args.batch_size,
    #         gamma=args.gamma,
    #         beta=args.beta,
    #         max_episodes=args.max_episodes
    #     )
  
    
    
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
            sem_inf_tick = posix_ipc.Semaphore("/sem_inf_tick")
            break
        except posix_ipc.ExistentialError:
            print("[INF] waiting for /sem_env_tick semaphore...")
            time.sleep(0.1)
 
    while True:
        try:
            sem_inf_done = posix_ipc.Semaphore("/sem_inf_done")
            break
        except posix_ipc.ExistentialError:
            print("[INF] waiting for /sem_inf_done semaphore...")
            time.sleep(0.1)


    header_fmt = "<d?q"
    header_size = struct.calcsize(header_fmt)
    vision_size = 60 #TODO unhardcode
    total_size = header_size + vision_size * 2


    mapfile = mmap.mmap(shm.fd, total_size)
    while True:

        sem_inf_tick.acquire()
        print("[INF] got signal")

        
        reward, game_over, action = struct.unpack_from(header_fmt, mapfile, 0)
        print(f"[Clock]{reward=},{game_over=},{action=}")
        # читаем vision как np.int8
        vision = np.frombuffer(mapfile, dtype=np.int8,
                       count=vision_size, offset=header_size)
        

        action_offset = struct.calcsize("<d?") 

        import random

        action = random.choice([0,1,2])

        struct.pack_into("q", mapfile, action_offset, action)
        print("[INF] wrote action")
        # сигналим Clock, что данные готовы
        sem_inf_done.release()

    shm.close_fd()