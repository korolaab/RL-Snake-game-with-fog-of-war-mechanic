import time
import posix_ipc
import argparse
import struct
import mmap
import mlflow
import signal
import sys

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


# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    print(f"\n[Clock] Received signal {signum}. Initiating graceful shutdown...")
    running = False

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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

print("[Clock] started. Press Ctrl+C to stop gracefully.")

mapfile = mmap.mmap(shm.fd, shm.size)

episode = 0
frame = 0
mlflow_run = None

def cleanup_resources():
    """Clean up all IPC resources and MLflow"""
    global mlflow_run
    print("[Clock] Cleaning up resources...")
    
    try:
        # End MLflow run gracefully
        if mlflow_run is not None:
            print("[Clock] Ending MLflow run...")
            mlflow.end_run()
            print("[Clock] MLflow run ended successfully")
        
        # Close memory mapped files
        mapfile.close()
        mapfile_env_ctrl.close()
        mapfile_inf_ctrl.close()
        
        # Close file descriptors
        shm.close_fd()
        sem_env_control.close_fd()
        sem_inf_control.close_fd()
        
        # Unlink semaphores
        sem_env_tick.unlink()
        sem_env_done.unlink()
        sem_inf_tick.unlink()
        sem_inf_done.unlink()
        sem_inf_training.unlink()
        
        # Unlink shared memory
        posix_ipc.unlink_shared_memory("/game_state")
        posix_ipc.unlink_shared_memory("/env_control")
        posix_ipc.unlink_shared_memory("/inf_control")
        
        print("[Clock] Cleanup completed.")
    except Exception as e:
        print(f"[Clock] Error during cleanup: {e}")

try:
    mlflow_run = mlflow.start_run()
    with mlflow_run:
        while running:
            if args.fps != 0:
                time.sleep(1.0 / args.fps)
            
            # Check running flag after sleep
            if not running:
                break
            
            reward, game_over, action = struct.unpack_from(header_fmt, mapfile, 0)
            if game_over == 1:
                episode += 1
                
                # reset header
                struct.pack_into("=i", mapfile_env_ctrl, 0, 1)
                struct.pack_into("=i", mapfile_inf_ctrl, 0, 1)

                sem_env_tick.release()   # разрешаем ENV работать
                sem_env_done.acquire()   # ждем, пока ENV скажет "готово"
                reset, snake_length, frames, sum_reward = struct.unpack_from(ctrl_fmt_env, mapfile_env_ctrl, 0)

                sem_inf_tick.release()   # разрешаем INF работать
                sem_inf_done.acquire()   # ждем, пока INF закончит
                
                do_train, loss, entropy_mean, loss_1_sum, entropy_sum = struct.unpack_from(ctrl_fmt_inf, mapfile_inf_ctrl, 0)
                print(f"[Clock] {episode}:{snake_length=} {loss=:0.3f} {entropy_mean=:0.3f} {loss_1_sum=:0.3f} {entropy_sum=:0.3f} {frames=}  {sum_reward=}")
                
                mlflow.log_metric("snake_length", snake_length, step=episode)
                mlflow.log_metric("loss", loss, step=episode)
                mlflow.log_metric("entropy_mean", entropy_mean, step=episode)
                mlflow.log_metric("loss_1_sum", loss_1_sum, step=episode)
                mlflow.log_metric("entropy_sum", entropy_sum, step=episode)
                mlflow.log_metric("frames", frames, step=episode)
                mlflow.log_metric("sum_reward", sum_reward, step=episode)

            else:
                sem_env_tick.release()   # разрешаем ENV работать
                sem_env_done.acquire()   # ждем, пока ENV скажет "готово"

                sem_inf_tick.release()   # разрешаем INF работать
                sem_inf_done.acquire()   # ждем, пока INF закончит

except KeyboardInterrupt:
    print("\n[Clock] Interrupted by user")
except Exception as e:
    print(f"[Clock] Error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("[Clock] Shutting down...")
    cleanup_resources()
    print("[Clock] Service stopped gracefully.")
    sys.exit(0)