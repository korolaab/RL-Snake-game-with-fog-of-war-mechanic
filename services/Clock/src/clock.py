import time
import posix_ipc
import argparse
import struct
import mmap
import mlflow

# Parse command line arguments
parser = argparse.ArgumentParser(description="Clock service for Snake RL")
parser.add_argument("--vision-size", type=int, default=5, help="Vision area size in bytes")
parser.add_argument("--fps", type=float, default=1.0, help="Frames per second (sleep interval)")
parser.add_argument('--mlflow_server', type=str, default=None, help='Mlflow server host')
parser.add_argument('--mlflow_experiment_name', required=True, type=str, help='Mlflow experiment')
args = parser.parse_args()

mlflow.set_tracking_uri(uri=args.mlflow_server)
mlflow.set_experiment(args.mlflow_experiment_name)
def manhattan_cells_without_center(R: int) -> int:
    return 2 * R * (R + 1)


header_fmt = "<d?q"
header_size = struct.calcsize(header_fmt)
vision_size = manhattan_cells_without_center(args.vision_size) + 2
total_size = header_size + (vision_size*8) * 2


# создаем shared memory

shm = posix_ipc.SharedMemory("/game_state", posix_ipc.O_CREX, size=total_size)

# создаем семафоры (Clock контролирует цикл)

sem_env_tick = posix_ipc.Semaphore("/sem_env_tick", posix_ipc.O_CREX, initial_value=0)
sem_env_done = posix_ipc.Semaphore("/sem_env_done", posix_ipc.O_CREX, initial_value=0)
# Control format: command (int)
# 0 = normal step
# 1 = reset
ctrl_fmt_env = "=iiii" # reset, snake_length, frames, sum_reward
sem_env_control = posix_ipc.SharedMemory("/env_control", posix_ipc.O_CREX, 
                size=struct.calcsize(ctrl_fmt_env))
mapfile_env_ctrl = mmap.mmap(sem_env_control.fd, sem_env_control.size)


sem_inf_tick = posix_ipc.Semaphore("/sem_inf_tick", posix_ipc.O_CREX, initial_value=0)
sem_inf_done = posix_ipc.Semaphore("/sem_inf_done", posix_ipc.O_CREX, initial_value=0)
sem_inf_training = posix_ipc.Semaphore("/sem_inf_training", posix_ipc.O_CREX, initial_value=0)

# Control format: command (int)
# 0 = normal step
# 1 = train
ctrl_fmt_inf = "=idddd" # do_train, loss, entropy.mean(), entropy.sum(). loss_1.sum()
sem_inf_control = posix_ipc.SharedMemory("/inf_control", posix_ipc.O_CREX, 
    size=struct.calcsize(ctrl_fmt_inf))
mapfile_inf_ctrl = mmap.mmap(sem_inf_control.fd, sem_inf_control.size)

print("[Clock] started")

mapfile = mmap.mmap(shm.fd, shm.size)

episode = 0
frame = 0
with mlflow.start_run():

    while True:
        if args.fps != 0:
            time.sleep(1.0 / args.fps)
        
        reward, game_over, action = struct.unpack_from(header_fmt, mapfile, 0)
        if game_over == 1:
            episode += 1
            
            
            # reset header
            struct.pack_into("=i", mapfile_env_ctrl, 0, 1)
            struct.pack_into("=i", mapfile_inf_ctrl, 0, 1)
            #frame = 0


            sem_env_tick.release()   # разрешаем ENV работать
            sem_env_done.acquire()   # ждем, пока ENV скажет "готово"
            reset, snake_length, frames, sum_reward = struct.unpack_from(ctrl_fmt_env, mapfile_env_ctrl, 0)

            sem_inf_tick.release()   # разрешаем INF работать
            sem_inf_done.acquire()   # ждем, пока INF закончит
            

            do_train, loss, entropy_mean, loss_1_sum, entropy_sum = struct.unpack_from(ctrl_fmt_inf, mapfile_inf_ctrl, 0)
            #print(f"[Clock] Episode {episode} done")
            print(f"[Clock] {episode}:{snake_length=} {loss=:0.3f} {entropy_mean=:0.3f} {loss_1_sum=:0.3f} {entropy_sum=:0.3f} {frames=}  {sum_reward=}")
            mlflow.log_metric("snake_length", snake_length,step=episode)
            mlflow.log_metric("loss", loss,step=episode)
            mlflow.log_metric("entropy_mean", entropy_mean,step=episode)
            mlflow.log_metric("loss_1_sum", loss_1_sum,step=episode)
            mlflow.log_metric("entropy_sum", entropy_sum,step=episode)
            mlflow.log_metric("frames", frames,step=episode)
            mlflow.log_metric("sum_reward", sum_reward,step=episode)

        else:
            #print(f"[Clock] {episode=} {frame=} {reward=} {game_over=} {action=}")
            #print("[Clock] tick → ENV")
            sem_env_tick.release()   # разрешаем ENV работать
            sem_env_done.acquire()   # ждем, пока ENV скажет "готово"

            #print("[Clock] ENV done")

            #print("[Clock] tick → INF")
            sem_inf_tick.release()   # разрешаем INF работать
            sem_inf_done.acquire()   # ждем, пока INF закончит
            
            #frame +=1
            #print("[Clock] step Done")
        #struct.pack_into(ctrl_fmt_env, mapfile_env_ctrl, 0, 0)
        #struct.pack_into(ctrl_fmt_inf, mapfile_inf_ctrl, 0, 0)


        
shm.close_fd()