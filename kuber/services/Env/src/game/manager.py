# game/manager.py

import threading
import random
import logging
from utils.seed import set_seed  
import socket
from .snake import SnakeGame

class GameManager:
    def __init__(self, grid_width, 
                       grid_height, 
                       vision_radius, 
                       vision_display_cols, 
                       vision_display_rows, 
                       fps, 
                       seed,
                       maxStepsWithoutApple,
                       reward_config,
                       max_snakes=10,
                       sync_enabled=False,
                       sync_host="sync_service_host",
                       sync_port=5555,
                       sync_buffer_size=1024):
        self.GRID_WIDTH = grid_width
        self.GRID_HEIGHT = grid_height
        self.VISION_RADIUS = vision_radius
        self.VISION_DISPLAY_COLS = vision_display_cols
        self.VISION_DISPLAY_ROWS = vision_display_rows
        self.FPS = fps
        self.MAX_SNAKES = max_snakes
        self.FOODS = set()
        self.snakes = {}
        self.snake_locks = {}
        self.GAME_OVER = False
        self.game_over_lock = threading.Lock()
        self.seed = seed 
        self.reward_config = reward_config
        self.maxStepsWithoutApple = maxStepsWithoutApple
        
        # UDP Synchronization configuration
        self.SYNC_ENABLED = sync_enabled
        self.SYNC_HOST = sync_host
        self.SYNC_PORT = sync_port
        self.SYNC_BUFFER_SIZE = sync_buffer_size
        self.sync_socket = None
        
        if self.SYNC_ENABLED:
            self._setup_sync_socket()
        
        set_seed(self.seed)
        self.game_over_raised = False
        threading.Thread(target=self.game_loop, daemon=True).start()
        
    def _setup_sync_socket(self):
        """Setup TCP socket for synchronization (client)"""
        import socket
        try:
            # TCP Client mode: connect to the sync server
            self.sync_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sync_socket.connect((self.SYNC_HOST, self.SYNC_PORT))
            self.sync_socket_file = self.sync_socket.makefile('rb')  # For easy readline
            logging.info({"event": "sync_setup", "status": "success (TCP)", "host": self.SYNC_HOST, "port": self.SYNC_PORT})
        except Exception as e:
            logging.error({"event": "sync_setup", "status": "error (TCP)", "error": str(e)})
            self.SYNC_ENABLED = False
            self.sync_socket = None

    def state(self):
        grid = {f"{x},{y}": [] for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)}
        for sid, game in self.snakes.items():
            with self.snake_locks[sid]:
                for i, p in enumerate(game.snake):
                    cell = f"{p[0]},{p[1]}"
                    typ = 'HEAD' if i == 0 else 'BODY'
                    grid[cell].append({'type': typ, 'snake_id': sid})
        for food in self.FOODS:
            cell = f"{food[0]},{food[1]}"
            grid[cell].append({'type': 'FOOD', 'snake_id': None})
        for cell, v in grid.items():
            if not v:
                grid[cell] = [{'type': 'EMPTY'}]
        visions = {sid: game.get_visible_cells() for sid, game in self.snakes.items()}
        with self.game_over_lock:
            global_game_over = self.GAME_OVER
        statuses = {sid: global_game_over for sid in self.snakes.keys()}
        return grid, visions, statuses, self.GAME_OVER

    def spawn_food(self):
        occupied = {pos for game in self.snakes.values() for pos in game.snake} | self.FOODS
        while True:
            pos = (random.randint(0, self.GRID_WIDTH - 1), random.randint(0, self.GRID_HEIGHT - 1))
            if pos not in occupied:
                self.FOODS.add(pos)
                break

    def find_safe_spawn_location(self):
        occupied = {pos for g in self.snakes.values() for pos in g.snake} | self.FOODS
        for _ in range(1000):
            head = (random.randrange(self.GRID_WIDTH), random.randrange(self.GRID_HEIGHT))
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                body = [(head[0] - i*dx, head[1] - i*dy) for i in range(3)]
                if all(0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT for x,y in body) and not any(pos in occupied for pos in body):
                    return body, (dx, dy)
        # fallback
        fallback = [(self.GRID_WIDTH//2 - i, self.GRID_HEIGHT//2) for i in range(3)]
        return fallback, (1, 0)

    def end_game_all(self):
        with self.game_over_lock:
            self.GAME_OVER = True

    def reset_game(self):
        with self.game_over_lock:
            self.GAME_OVER = True
        # Дать стримам завершиться
        import time
        time.sleep(0.1)
        with self.game_over_lock:
            self.GAME_OVER = False
        self.snakes.clear()
        self.snake_locks.clear()
        self.FOODS.clear()
        self.spawn_food()
        self.game_over_raised = False
        logging.info({"event": "game_reset", "action": "all_snakes_removed_food_respawned"})

    def add_snake(self, snake_id):
        if len(self.snakes) >= self.MAX_SNAKES:
            return False
        snake = SnakeGame(snake_id, self)
        self.snakes[snake_id] = snake
        self.snake_locks[snake_id] = threading.Lock()
        return True

    def remove_snake(self, snake_id):
        if snake_id in self.snakes:
            del self.snakes[snake_id]
        if snake_id in self.snake_locks:
            del self.snake_locks[snake_id]

    def get_snake(self, snake_id):
        return self.snakes.get(snake_id)

    def get_lock(self, snake_id):
        return self.snake_locks.get(snake_id)

    def connect_sync_socket(self):
        """Setup TCP socket for synchronization (client)"""
        import socket
        try:
            # Detailed logging before connection attempt
            logging.debug({
                "event": "sync_socket_connection_start", 
                "host": self.SYNC_HOST, 
                "port": self.SYNC_PORT,
                "current_time": str(datetime.now())
            })

            # TCP Client mode: connect to the sync server
            self.sync_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Set socket options for more robust connection
            self.sync_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sync_socket.settimeout(10)  # 10-second timeout for connection
            
            try:
                # Attempt connection
                self.sync_socket.connect((self.SYNC_HOST, self.SYNC_PORT))
                
                # Log successful connection details
                logging.debug({
                    "event": "sync_socket_connected", 
                    "local_address": self.sync_socket.getsockname(),
                    "remote_address": self.sync_socket.getpeername()
                })
                
                # Send identification with extra logging
                identification = b'ENV\n'
                logging.debug({
                    "event": "sending_client_identification", 
                    "identification_bytes": identification,
                    "identification_str": identification.decode().strip()
                })
                
                send_result = self.sync_socket.send(identification)
                logging.debug({
                    "event": "client_identification_sent", 
                    "bytes_sent": send_result
                })
                
                # Reset timeout for subsequent operations
                self.sync_socket.settimeout(None)
                
                # Create file-like object for easier reading
                self.sync_socket_file = self.sync_socket.makefile('rb')
                
                logging.info({
                    "event": "sync_setup", 
                    "status": "success (TCP)", 
                    "host": self.SYNC_HOST, 
                    "port": self.SYNC_PORT,
                    "connection_details": {
                        "local_address": self.sync_socket.getsockname(),
                        "remote_address": self.sync_socket.getpeername()
                    }
                })
            
            except (socket.timeout, socket.error) as conn_error:
                logging.error({
                    "event": "sync_connection_error", 
                    "error": str(conn_error),
                    "error_type": type(conn_error).__name__,
                    "host": self.SYNC_HOST, 
                    "port": self.SYNC_PORT
                })
                raise
        
        except Exception as e:
            logging.error({
                "event": "sync_setup", 
                "status": "error (TCP)", 
                "error": str(e),
                "error_type": type(e).__name__,
                "host": self.SYNC_HOST, 
                "port": self.SYNC_PORT
            })
            self.SYNC_ENABLED = False
            self.sync_socket = None
            self.sync_socket_file = None
        import socket
        try:
            self.sync_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sync_socket.connect((self.SYNC_HOST, self.SYNC_PORT))
            # Identify as ENV
            self.sync_socket.sendall(b'ENV\n')
            self.sync_socket_file = self.sync_socket.makefile('rb')  # For easy readline
            logging.info({"event": "sync_reconnect", "status": "success (TCP)", "host": self.SYNC_HOST, "port": self.SYNC_PORT})
        except Exception as e:
            self.sync_socket = None
            self.sync_socket_file = None
            logging.error({"event": "tcp_reconnect_failed", "error": str(e)})

    def check_sync_socket(self):
        if self.sync_socket is None:
            self.connect_sync_socket()
        else:
            try:
                # Do not send ping to server; just check with empty bytes (triggers socket error if broken)
                self.sync_socket.sendall(b"")
            except Exception:
                self.connect_sync_socket()

    def game_loop(self):
        import time
        import logging
        import json
        
        self.reset_game()
        buffer = b''
        while True:
    # Wait for next step trigger (either external TCP sync or internal timer)
            if self.SYNC_ENABLED:
                self.check_sync_socket()  # << Ensure connected (including first time)
                if self.sync_socket and self.sync_socket_file:
                    try:
                        # Detailed sync process logging
                        logging.debug({
                            "event": "preparing_to_read_sync_signal",
                            "socket_status": {
                                "local_address": self.sync_socket.getsockname(),
                                "remote_address": self.sync_socket.getpeername()
                            }
                        })
                        
                        # Read with timeout and detailed logging
                        self.sync_socket.settimeout(10)  # 10-second timeout
                        line = self.sync_socket_file.readline()
                        
                        if not line:
                            logging.warning({
                                "event": "empty_sync_signal", 
                                "message": "Sync TCP connection seems closed"
                            })
                            raise Exception("Sync TCP connection closed")
                        
                        # Log received line details
                        logging.debug({
                            "event": "received_sync_signal_TCP", 
                            "raw_line": line,
                            "decoded_line": line.decode().strip() if line else None
                        })
                        
                        logging.info({
                            "event": "sync_signal_processed", 
                            "status": "success"
                        })
                    
                    except socket.timeout:
                        logging.warning({
                            "event": "sync_signal_timeout", 
                            "message": "Timeout waiting for sync signal"
                        })
                        self.sync_socket = None
                        self.sync_socket_file = None
                        continue
                    except Exception as e:
                        logging.error({
                            "event": "TCP_sync_error", 
                            "error": str(e),
                            "error_type": type(e).__name__
                        })
                        self.sync_socket = None
                        self.sync_socket_file = None
                        continue  # Try to recover next loop
                else:
                    logging.debug({
                        "event": "sync_socket_not_ready", 
                        "sync_socket": self.sync_socket is not None,
                        "sync_socket_file": self.sync_socket_file is not None
                    })
            else:
                # Internal timing based on FPS
                time.sleep(1.0 / self.FPS)
            
            # Process game state updates
            for sid, game in list(self.snakes.items()):
                with self.snake_locks[sid]:
                    status = game.update(self.GAME_OVER)
                    if status == 'collision' or status == 'starvation':
                        self.GAME_OVER = True
                        logging.info({"event": "game_over", "reason": status, "snake_id": sid})

                    if self.GAME_OVER != True:
                        grid, visions, statuses, game_over = self.state()
                        logging.info({"grid": grid,
                                    "visions": visions,
                                    "statuses": statuses,
                                    "game_over": game_over})
                    elif self.game_over_raised == False:
                        snake_lens = {}
                        for sid, game in list(self.snakes.items()):
                            with self.snake_locks[sid]:
                                snake_len = len(game.snake)
                            snake_lens[sid] = snake_len

                        logging.info({"event": "game_over_results",
                                "snakes_lengths": snake_lens
                                })
                        self.game_over_raised = True

                    if len(self.FOODS) == 0:
                        self.spawn_food()

            # ---- SEND ready at END of LOOP ----
            if self.SYNC_ENABLED:
                if self.sync_socket:
                    try:
                        self.sync_socket.sendall(b"ready\n")
                        logging.debug({"event": "sent_ready_TCP"})
                    except Exception as e:
                        logging.error({"event": f"tcp_send_ready_error: {e}"})
                        self.sync_socket = None