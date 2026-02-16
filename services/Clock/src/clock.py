import time
import posix_ipc
import argparse
import struct
import mmap
import mlflow
import signal
import sys
import glob
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description="Clock service for Snake RL")
parser.add_argument("--vision-size", type=int, default=5, help="Vision area size in bytes")
parser.add_argument("--fps", type=float, default=1.0, help="Frames per second (sleep interval)")
parser.add_argument('--mlflow_server', type=str, default=None, help='Mlflow server host')
parser.add_argument('--mlflow_experiment_name', required=True, type=str, help='Mlflow experiment')
parser.add_argument("--max-episodes", type=int, default=None, help="Maximum episodes to run")
parser.add_argument("--apple-speed", type=float, default=0, help="Apple movement speed (for MLflow logging)")
parser.add_argument("--num-snakes", type=int, default=1, help="Number of snakes (1 or 2)")
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


if args.num_snakes == 2:
    header_fmt = "<d?qq"
else:
    header_fmt = "<d?q"
header_size = struct.calcsize(header_fmt)
vision_size = manhattan_cells_without_center(args.vision_size) + 2
state_bytes = (vision_size * 8) * 2  # one state buffer in bytes
total_size = header_size + state_bytes * args.num_snakes


# создаем shared memory
shm = posix_ipc.SharedMemory("/game_state", posix_ipc.O_CREX, size=total_size)

# создаем семафоры (Clock контролирует цикл)
sem_env_tick = posix_ipc.Semaphore("/sem_env_tick", posix_ipc.O_CREX, initial_value=0)
sem_env_done = posix_ipc.Semaphore("/sem_env_done", posix_ipc.O_CREX, initial_value=0)

# Control format: command (int)
# 0 = normal step
# 1 = reset
ctrl_fmt_env = "=iiiii" # reset, snake_length, eaten_apples, frames, sum_reward
sem_env_control = posix_ipc.SharedMemory("/env_control", posix_ipc.O_CREX, 
                size=struct.calcsize(ctrl_fmt_env))
mapfile_env_ctrl = mmap.mmap(sem_env_control.fd, sem_env_control.size)


sem_inf_tick = posix_ipc.Semaphore("/sem_inf_tick", posix_ipc.O_CREX, initial_value=0)
sem_inf_done = posix_ipc.Semaphore("/sem_inf_done", posix_ipc.O_CREX, initial_value=0)
sem_inf_training = posix_ipc.Semaphore("/sem_inf_training", posix_ipc.O_CREX, initial_value=0)

# Control format: command (int)
# 0 = normal step
# 1 = train
ctrl_fmt_inf = "=idddddddddd" # do_train, loss, policy_loss, entropy_mean, entropy_std, grad_norm, returns_mean, returns_std, action_0_freq, action_1_freq, action_2_freq
sem_inf_control = posix_ipc.SharedMemory("/inf_control", posix_ipc.O_CREX, 
    size=struct.calcsize(ctrl_fmt_inf))
mapfile_inf_ctrl = mmap.mmap(sem_inf_control.fd, sem_inf_control.size)

print("[Clock] started. Press Ctrl+C to stop gracefully.")

mapfile = mmap.mmap(shm.fd, shm.size)

episode = 0
frame = 0
mlflow_run = None

# Track moving averages
reward_history = []
length_history = []
apples_history = []
snake_length_history = []

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
        # Log hyperparameters once at start
        mlflow.log_params({
            "vision_size": args.vision_size,
            "fps": args.fps,
            "mlflow_experiment_name": args.mlflow_experiment_name,
            "architecture": "REINFORCE",
            "shared_memory_communication": True,
            "apple_speed": args.apple_speed,
            "num_snakes": args.num_snakes
        })
        while running and (args.max_episodes is None or episode < args.max_episodes):
            if args.fps != 0:
                time.sleep(1.0 / args.fps)
            
            # Check running flag after sleep
            if not running:
                break
            
            header_data = struct.unpack_from(header_fmt, mapfile, 0)
            reward = header_data[0]
            game_over = header_data[1]
            if game_over == 1:
                episode += 1
                
                # reset header
                struct.pack_into("=i", mapfile_env_ctrl, 0, 1)
                struct.pack_into("=i", mapfile_inf_ctrl, 0, 1)

                sem_env_tick.release()   # разрешаем ENV работать
                sem_env_done.acquire()   # ждем, пока ENV скажет "готово"
                reset, snake_length, eaten_apples, frames, sum_reward = struct.unpack_from(ctrl_fmt_env, mapfile_env_ctrl, 0)

                sem_inf_tick.release()   # разрешаем INF работать
                sem_inf_done.acquire()   # ждем, пока INF закончит
                
                do_train, loss, policy_loss, entropy_mean, entropy_std, grad_norm, returns_mean, returns_std, action_0_freq, action_1_freq, action_2_freq = struct.unpack_from(ctrl_fmt_inf, mapfile_inf_ctrl, 0)
                print(f"[Clock] {episode}:{snake_length=} {eaten_apples=} {loss=:0.3f} {policy_loss=:0.3f} {entropy_mean=:0.3f} {grad_norm=:0.3f} {frames=}  {sum_reward=}")
                
                # Update moving averages
                reward_history.append(sum_reward)
                length_history.append(frames)
                apples_history.append(eaten_apples)
                snake_length_history.append(snake_length)
                
                # Prepare metrics dict
                metrics = {
                    "episode_reward": sum_reward,
                    "episode_length": frames,
                    "eaten_apples": eaten_apples,
                    "snake_length": snake_length,
                    "total_loss": loss,
                    "policy_loss": policy_loss,
                    "entropy_mean": entropy_mean,
                    "entropy_std": entropy_std,
                    "grad_norm": grad_norm,
                    "returns_mean": returns_mean,
                    "returns_std": returns_std,
                    "action_0_freq": action_0_freq,
                    "action_1_freq": action_1_freq,
                    "action_2_freq": action_2_freq
                }
                
                # Add moving averages (last 100 episodes)
                if len(reward_history) >= 100:
                    metrics["reward_100ep_avg"] = sum(reward_history[-100:]) / 100
                    metrics["length_100ep_avg"] = sum(length_history[-100:]) / 100
                    metrics["apples_100ep_avg"] = sum(apples_history[-100:]) / 100
                    metrics["snake_length_100ep_avg"] = sum(snake_length_history[-100:]) / 100
                elif len(reward_history) >= 10:
                    # Use available history if less than 100 episodes
                    metrics["reward_10ep_avg"] = sum(reward_history[-10:]) / len(reward_history[-10:])
                    metrics["length_10ep_avg"] = sum(length_history[-10:]) / len(length_history[-10:])
                    metrics["apples_10ep_avg"] = sum(apples_history[-10:]) / len(apples_history[-10:])
                    metrics["snake_length_10ep_avg"] = sum(snake_length_history[-10:]) / len(snake_length_history[-10:])
                
                # Log comprehensive metrics
                mlflow.log_metrics(metrics, step=episode)
                
                # Model checkpointing every 50 episodes
                if episode % 50 == 0:
                    import glob
                    import os
                    model_files = glob.glob("/logs/model_checkpoint_*.pth")
                    if model_files:
                        # Log latest model as artifact
                        latest_model = max(model_files, key=os.path.getctime)
                        mlflow.log_artifact(latest_model, "models")
                        print(f"[Clock] Logged model checkpoint: {os.path.basename(latest_model)}")
                        # Clean up after logging
                        os.remove(latest_model)

            else:
                sem_env_tick.release()   # разрешаем ENV работать
                sem_env_done.acquire()   # ждем, пока ENV скажет "готово"

                sem_inf_tick.release()   # разрешаем INF работать
                sem_inf_done.acquire()   # ждем, пока INF закончит
        
        # Check if we reached max episodes
        if args.max_episodes is not None and episode >= args.max_episodes:
            print(f"[Clock] Reached maximum episodes ({args.max_episodes}). Stopping experiment.")

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