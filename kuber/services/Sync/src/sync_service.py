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
        logger.info({"message": "Starting sync service (TCP)"})
        signal_count = 0
        try:
            while True:
                start_time = time.time()
                self.accept_clients()
                signal_count += 1
                timestamp = datetime.now().isoformat()
                signal_data = json.dumps({
                    "timestamp": timestamp,
                    "sequence": signal_count
                }).encode('utf-8') + b'\n'  # newline as delimiter

                # Send message to all clients, drop on error
                disconnected = []
                for sock in self.client_sockets:
                    try:
                        sock.sendall(signal_data)
                    except Exception as e:
                        logger.warning({"event": "client_disconnected", "error": str(e)})
                        disconnected.append(sock)
                for dsock in disconnected:
                    self.client_sockets.remove(dsock)
                    dsock.close()

                logger.debug({
                    "event": "Sent sync signal (TCP)",
                    "signal_number": signal_count
                })

                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                if self.verbose and sleep_time < 0:
                    logger.warning({
                        "message": "Sync signal took longer than interval",
                        "elapsed": round(elapsed, 4),
                        "interval": round(self.interval, 4)
                    })
                time.sleep(sleep_time)
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
    sync_service.run()
