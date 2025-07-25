#!/usr/bin/env python3
import argparse
import socket
import time
import sys
import json
import logging
from datetime import datetime
import logger # from utils/logger.py 

# Configure logging with our custom implementation
logger.setup()
logger = logging.getLogger("sync_service")

class SyncService:
    def __init__(self, host, port, frequency, verbose=False, filepath=None):
        self.host = host
        self.port = port
        self.interval = 1.0 / frequency
        self.verbose = verbose
        self.filepath = filepath
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Create TCP socket (server)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.setblocking(False)  # Non-blocking accept
        
        self.client_sockets = []
        logger.info({
            "event": "Sync TCP server initialized",
            "frequency": frequency,
            "interval": round(self.interval, 4),
            "bind_address": f"{host}:{port}"
        })

    def accept_clients(self):
        try:
            while True:
                try:
                    client_sock, addr = self.server_socket.accept()
                    client_sock.setblocking(False)
                    self.client_sockets.append(client_sock)
                    logger.info({"event": "client_connected", "addr": addr})
                except BlockingIOError:
                    break
        except Exception as e:
            logger.error({"event": "accept_clients_error", "error": str(e)})

    def run(self):
        logger.info({"message": "Starting sync service (TCP; algorithm: wait for ready packets)"})
        signal_count = 0

        def recv_ready(sock):
            try:
                data = b""
                while not data.endswith(b'\n'):
                    chunk = sock.recv(1024)
                    if not chunk:
                        raise Exception("Client closed connection while waiting for ready")
                    data += chunk
                msg = data.decode().strip()
                logger.debug({"event": "recv_ready", "from": str(sock.getpeername()), "msg": msg})
                return msg == "ready"
            except Exception as e:
                logger.warning({"event": "error_waiting_for_ready", "error": str(e)})
                return False

        try:
            while True:
                self.accept_clients()

                if len(self.client_sockets) < 2:
                    logger.info("Waiting for both Env and Inference to connect...")
                    time.sleep(1)
                    continue

                env_sock, inf_sock = self.client_sockets[:2]
                signal_count += 1
                timestamp = datetime.now().isoformat()
                signal_data = json.dumps({
                    "timestamp": timestamp,
                    "sequence": signal_count
                }).encode('utf-8') + b'\n'

                # [1] Send sync to Env, wait for ready
                try:
                    env_sock.sendall(signal_data)
                    logger.debug({"event": "Sent sync to Env", "signal_number": signal_count})
                    if not recv_ready(env_sock):
                        raise Exception("Did not receive ready from Env")
                except Exception as e:
                    logger.warning({"event": "env_disconnected", "error": str(e)})
                    self.client_sockets.remove(env_sock)
                    env_sock.close()
                    continue

                # [2] Send sync to Inference, wait for ready
                try:
                    inf_sock.sendall(signal_data)
                    logger.debug({"event": "Sent sync to Inf", "signal_number": signal_count})
                    if not recv_ready(inf_sock):
                        raise Exception("Did not receive ready from Inf")
                except Exception as e:
                    logger.warning({"event": "inference_disconnected", "error": str(e)})
                    self.client_sockets.remove(inf_sock)
                    inf_sock.close()
                    continue

                # [3] Control timing if needed
                time.sleep(max(0, self.interval - (time.time() - datetime.fromisoformat(timestamp).timestamp())))
        except KeyboardInterrupt:
            logger.info({"message": "Sync service stopped by user"})
        except Exception as e:
            logger.error({
                "message": "Error in sync service (TCP)",
                "error": str(e)
            })
        finally:
            self.server_socket.close()
            for sock in self.client_sockets:
                sock.close()
            logger.info({"message": "Sync service shut down"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCP Synchronization Service for ML System")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Address to bind TCP server")
    parser.add_argument("--port", type=int, default=5555, help="TCP port for synchronization")
    parser.add_argument("--frequency", type=float, default=10.0, help="Sync frequency in Hz (default: 10Hz)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    sync_service = SyncService(
        host=args.host,
        port=args.port,
        frequency=args.frequency,
        verbose=args.verbose
    )
    sync_service.run()
