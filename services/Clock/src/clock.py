import time
import posix_ipc
import argparse
import struct
import mmap
# Parse command line arguments
parser = argparse.ArgumentParser(description="Clock service for Snake RL")
parser.add_argument("--vision-size", type=int, default=5, help="Vision area size in bytes")
parser.add_argument("--fps", type=float, default=1.0, help="Frames per second (sleep interval)")
args = parser.parse_args()

def manhattan_cells_without_center(R: int) -> int:
    return 2 * R * (R + 1)


header_fmt = "<d?q"
header_size = struct.calcsize(header_fmt)
vision_size = manhattan_cells_without_center(args.vision_size)
total_size = header_size + vision_size * 2


# создаем shared memory

shm = posix_ipc.SharedMemory("/game_state", posix_ipc.O_CREX, size=total_size)

# создаем семафоры (Clock контролирует цикл)

sem_env_tick = posix_ipc.Semaphore("/sem_env_tick", posix_ipc.O_CREX, initial_value=0)
sem_env_done = posix_ipc.Semaphore("/sem_env_done", posix_ipc.O_CREX, initial_value=0)
# Control format: command (int)
# 0 = normal step
# 1 = reset
ctrl_fmt_env = "=i"
sem_env_control = posix_ipc.SharedMemory("/env_control", posix_ipc.O_CREX, size=64)
mapfile_env_ctrl = mmap.mmap(sem_env_control.fd, sem_env_control.size)


sem_inf_tick = posix_ipc.Semaphore("/sem_inf_tick", posix_ipc.O_CREX, initial_value=0)
sem_inf_done = posix_ipc.Semaphore("/sem_inf_done", posix_ipc.O_CREX, initial_value=0)
sem_inf_training = posix_ipc.Semaphore("/sem_inf_training", posix_ipc.O_CREX, initial_value=0)

# Control format: command (int)
# 0 = normal step
# 1 = train
ctrl_fmt_inf = "=i"
sem_inf_control = posix_ipc.SharedMemory("/inf_control", posix_ipc.O_CREX, size=64)
mapfile_inf_ctrl = mmap.mmap(sem_inf_control.fd, sem_inf_control.size)

print("[Clock] started")

mapfile = mmap.mmap(shm.fd, shm.size)

episode = 0
frame = 0
while True:
    if args.fps != 0:
        time.sleep(1.0 / args.fps)
    
    reward, game_over, action = struct.unpack_from(header_fmt, mapfile, 0)
    if game_over == 1:
        episode += 1
        print(f"[Clock] Episode {episode} done")
        # reset header
        struct.pack_into(ctrl_fmt_env, mapfile_env_ctrl, 0, 1)
        struct.pack_into(ctrl_fmt_inf, mapfile_inf_ctrl, 0, 1)
        frame = 0
        print("[Clock] Env reset")


    print(f"[Clock] {frame=} {reward=},{game_over=},{action=}")
    print("[Clock] tick → ENV")
    sem_env_tick.release()   # разрешаем ENV работать

    sem_env_done.acquire()   # ждем, пока ENV скажет "готово"
    struct.pack_into(ctrl_fmt_env, mapfile_env_ctrl, 0, 0)
    struct.pack_into(ctrl_fmt_inf, mapfile_inf_ctrl, 0, 0)
    print("[Clock] ENV done")




    print("[Clock] tick → INF")
    sem_inf_tick.release()   # разрешаем INF работать
    sem_inf_done.acquire()   # ждем, пока INF закончит
    

    print("[Clock] step Done")
    frame +=1
shm.close_fd()