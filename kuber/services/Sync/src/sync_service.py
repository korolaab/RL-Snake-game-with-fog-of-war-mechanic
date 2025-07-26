#!/usr/bin/env python3
import argparse
import socket
import time
import sys
import json
import logging
from datetime import datetime
import select
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

        # Initialize client socket attributes
        self.inf_sock = None
        self.env_sock = None
        self.client_sockets = []
        logger.info({
            "event": "Sync TCP server initialized",
            "frequency": frequency,
            "interval": round(self.interval, 4),
            "bind_address": f"{host}:{port}"
        })

    def accept_clients(self):
        """Accept connections, assign to ENV/INF slot as soon as available."""
        self.server_socket.setblocking(False)
        try:
            client_sock, addr = self.server_socket.accept()
            client_sock.settimeout(15)  # Extended timeout for identification
            
            try:
                # More robust client identification
                client_id = b''
                start_time = time.time()
                while len(client_id) < 16 and time.time() - start_time < 15:
                    try:
                        chunk = client_sock.recv(1)
                    except socket.timeout:
                        chunk = b''
                    
                    if not chunk:
                        break
                    client_id += chunk
                    
                    # Convert to string for more flexible matching
                    id_str = client_id.decode(errors='ignore').strip()
                    
                    # More flexible matching
                    if id_str.startswith('INF') or id_str.startswith('ENV'):
                        break
                
                # Trim any additional data
                client_id = client_id.strip()
                
                logger.debug({
                    "event": "client_connection_attempt", 
                    "received_bytes": client_id,
                    "decoded_str": id_str,
                    "addr": str(addr)
                })
                
                # More flexible identification
                if b'INF' in client_id or id_str.startswith('INF'):
                    # Close previous INF connection if exists
                    if hasattr(self, 'inf_sock') and self.inf_sock:
                        try:
                            self.inf_sock.close()
                        except Exception:
                            pass
                    
                    self.inf_sock = client_sock
                    logger.info({
                        "event": "client_connected", 
                        "type": "INF", 
                        "addr": str(addr),
                        "identification_detail": id_str
                    })
                
                elif b'ENV' in client_id or id_str.startswith('ENV'):
                    # Close previous ENV connection if exists
                    if hasattr(self, 'env_sock') and self.env_sock:
                        try:
                            self.env_sock.close()
                        except Exception:
                            pass
                    
                    self.env_sock = client_sock
                    logger.info({
                        "event": "client_connected", 
                        "type": "ENV", 
                        "addr": str(addr),
                        "identification_detail": id_str
                    })
                
                else:
                    logger.warning({
                        "event": "unknown_client", 
                        "received_data": client_id, 
                        "addr": str(addr),
                        "warning": "Unrecognized client type"
                    })
                    client_sock.close()
            
            except (socket.timeout, BlockingIOError, socket.error) as e:
                logger.warning({
                    "event": "client_identification_failure", 
                    "error": str(e), 
                    "addr": str(addr),
                    "error_type": type(e).__name__,
                    "received_partial_data": client_id
                })
                try:
                    client_sock.close()
                except Exception:
                    pass
        
        except BlockingIOError:
            # No client to accept right now, which is fine
            pass
        except Exception as e:
            logger.error({
                "event": "accept_clients_error", 
                "error": str(e),
                "error_type": type(e).__name__
            })

    def _check_client_alive(self, sock):
        """Return False if socket closed/disconnected, else True."""
        try:
            # Use select to see if the socket is readable
            ready, _, _ = select.select([sock], [], [], 0)
            if ready:
                data = sock.recv(1, socket.MSG_PEEK)
                if not data:
                    return False
            return True
        except Exception:
            return False

    def run(self):
        logger.info({"message": "Starting sync service (TCP; algorithm: wait for ready packets)"})
        logger.debug({"event": "sync_service_run_starting"})
        signal_count = 0

        def recv_ready(sock):
            """
            More aggressive ready message handling
            - Explicitly send a trigger to client
            - More flexible message parsing
            - Detailed logging
            """
            try:
                # Longer timeout for receiving 'ready'
                sock.settimeout(15)  # 15 seconds timeout
                logger.debug({"event": "recv_ready_start", "from": str(sock.getpeername())})
                
                # Send a trigger to encourage client to respond
                try:
                    sock.sendall(b'sync_ready_check\n')
                    logger.debug({
                        "event": "sent_ready_trigger", 
                        "to": str(sock.getpeername())
                    })
                except Exception as trigger_error:
                    logger.warning({
                        "event": "failed_to_send_ready_trigger", 
                        "error": str(trigger_error)
                    })
                
                # Read and process messages more flexibly
                total_buffer = b''
                start_time = time.time()
                
                while time.time() - start_time < 15:  # Overall 15-second timeout
                    try:
                        # Read in smaller, more frequent chunks
                        data = sock.recv(256)
                        
                        if not data:
                            logger.debug({"event": "client_closed", "from": str(sock.getpeername())})
                            return False
                        
                        total_buffer += data
                        
                        # Try to decode the entire buffer
                        try:
                            msg = total_buffer.decode(errors='ignore').strip().lower()
                            
                            logger.debug({
                                "event": "recv_message", 
                                "from": str(sock.getpeername()), 
                                "msg": msg,
                                "buffer_length": len(total_buffer)
                            })
                            
                            # More flexible ready detection
                            if ("ready" in msg) or ("ok" in msg) or ("sync" in msg):
                                logger.info({
                                    "event": "received_ready", 
                                    "from": str(sock.getpeername()),
                                    "matched_msg": msg
                                })
                                return True
                            
                            # Reset buffer if it gets too large
                            if len(total_buffer) > 4096:
                                total_buffer = total_buffer[-4096:]
                        
                        except Exception as decode_error:
                            # Log decode errors but continue
                            logger.debug({
                                "event": "decode_error", 
                                "error": str(decode_error),
                                "buffer": total_buffer
                            })
                            continue
                    
                    except (socket.timeout, BlockingIOError):
                        # Just continue if no data is immediately available
                        time.sleep(0.1)
                        continue
                
                # Timeout or no ready message
                logger.warning({
                    "event": "no_ready_received", 
                    "from": str(sock.getpeername()),
                    "total_buffer": total_buffer,
                    "timeout_duration": time.time() - start_time
                })
                return False
            
            except Exception as e:
                logger.error({
                    "event": "critical_error_waiting_for_ready", 
                    "error": str(e), 
                    "from": str(sock.getpeername()),
                    "error_type": type(e).__name__
                })
                return False
            try:
                # Extended timeout and more flexible ready detection
                sock.settimeout(15)  # 15 seconds timeout for receiving 'ready'
                logger.debug({"event": "recv_ready_start", "from": str(sock.getpeername())})
                
                # Read and process messages more flexibly
                total_buffer = b''
                start_time = time.time()
                while time.time() - start_time < 15:  # 15-second overall timeout
                    try:
                        # Read in smaller chunks to be more responsive
                        data = sock.recv(256)
                        if not data:
                            logger.debug({"event": "client_closed", "from": str(sock.getpeername())})
                            return False
                        
                        total_buffer += data
                        # Try to decode the entire buffer
                        try:
                            msg = total_buffer.decode().strip()
                            logger.debug({
                                "event": "recv_message", 
                                "from": str(sock.getpeername()), 
                                "msg": msg,
                                "buffer_length": len(total_buffer)
                            })
                            
                            # More flexible ready detection
                            if "ready" in msg.lower():
                                logger.debug({
                                    "event": "received_ready", 
                                    "from": str(sock.getpeername()),
                                    "matched_msg": msg
                                })
                                return True
                            
                            # Reset buffer if it gets too large
                            if len(total_buffer) > 4096:
                                total_buffer = total_buffer[-4096:]
                        
                        except UnicodeDecodeError:
                            # If decoding fails, we might have a partial message
                            continue
                    
                    except (socket.timeout, BlockingIOError):
                        # Just continue if no data is immediately available
                        time.sleep(0.1)
                        continue
                
                logger.warning({
                    "event": "no_ready_received", 
                    "from": str(sock.getpeername()),
                    "total_buffer": total_buffer
                })
                return False
            
            except Exception as e:
                logger.error({
                    "event": "critical_error_waiting_for_ready", 
                    "error": str(e), 
                    "from": str(sock.getpeername()),
                    "error_type": type(e).__name__
                })
                return False

        last_synced = 'INF'  # so ENV gets first
        try:
            while True:
                self.accept_clients()
                logger.debug({"event": "accepted_clients", "env_connected": hasattr(self, 'env_sock') and self.env_sock is not None, "inf_connected": hasattr(self, 'inf_sock') and self.inf_sock is not None})
                # Check liveness, remove if disconnected# Enhanced client connection and disconnection handling
                env_was_connected = hasattr(self, 'env_sock') and self.env_sock is not None
                inf_was_connected = hasattr(self, 'inf_sock') and self.inf_sock is not None

                if env_was_connected and not self._check_client_alive(self.env_sock):
                    logger.warning({
                        "event": "env_sock_dead", 
                        "message": "Environment client disconnected unexpectedly"
                    })
                    try:
                        self.env_sock.close()
                    except Exception as e:
                        logger.error({"event": "env_sock_close_error", "error": str(e)})
                    self.env_sock = None

                if inf_was_connected and not self._check_client_alive(self.inf_sock):
                    logger.warning({
                        "event": "inf_sock_dead", 
                        "message": "Inference client disconnected unexpectedly"
                    })
                    try:
                        self.inf_sock.close()
                    except Exception as e:
                        logger.error({"event": "inf_sock_close_error", "error": str(e)})
                    self.inf_sock = None

                # Log state changes
                if env_was_connected and self.env_sock is None:
                    logger.info({"event": "env_client_disconnected"})
                if inf_was_connected and self.inf_sock is None:
                    logger.info({"event": "inf_client_disconnected"})

                # Alternating logic
                turn = None
                logger.debug({"event": "turn_decision_pre", "last_synced": last_synced})
                if last_synced == 'INF' and hasattr(self, 'env_sock') and self.env_sock is not None:
                    turn = 'ENV'
                elif last_synced == 'ENV' and hasattr(self, 'inf_sock') and self.inf_sock is not None:
                    turn = 'INF'
                elif hasattr(self, 'env_sock') and self.env_sock is not None:
                    turn = 'ENV'
                elif hasattr(self, 'inf_sock') and self.inf_sock is not None:
                    turn = 'INF'
                else:
                    logger.info({
                        "event": "waiting_for_clients",
                        "env_connected": hasattr(self, 'env_sock') and self.env_sock is not None,
                        "inf_connected": hasattr(self, 'inf_sock') and self.inf_sock is not None
                    })
                    time.sleep(0.5)
                    continue

                logger.debug({"event": "turn_decided", "turn": turn})

                signal_count += 1
                timestamp = datetime.now().isoformat()
                signal_data = json.dumps({
                    "timestamp": timestamp,
                    "sequence": signal_count
                }).encode('utf-8') + b'\n'

                sock_to_use = self.env_sock if turn == 'ENV' else self.inf_sock
                try:
                    logger.info({"event": f"Sent sync to {turn}", "signal_number": signal_count})
                    logger.debug({"event": "before_recv_ready", "turn": turn, "from": str(sock_to_use.getpeername())})
                    if not recv_ready(sock_to_use):
                        raise Exception(f"Did not receive ready from {turn}")
                    logger.debug({"event": "after_recv_ready", "turn": turn, "from": str(sock_to_use.getpeername())})
                    last_synced = turn
                except Exception as e:
                    logger.warning({"event": f"{turn.lower()}_disconnected", "error": str(e)})
                    try:
                        sock_to_use.close()
                    except Exception:
                        pass
                    if turn == 'ENV':
                        self.env_sock = None
                    else:
                        self.inf_sock = None
                    continue

                # Timing
                elapsed = time.time() - datetime.fromisoformat(timestamp).timestamp()
                logger.debug({"event": "post_sync_timing", "elapsed": elapsed, "interval": self.interval})
                time.sleep(max(0, self.interval - elapsed))
        except KeyboardInterrupt:
            logger.info({"message": "Sync service stopped by user"})
        except Exception as e:
            logger.error({
                "message": "Error in sync service (TCP)",
                "error": str(e)
            })
        finally:
            logger.debug({"event": "shutting_down"})
            self.server_socket.close()
            if hasattr(self, 'env_sock'):
                self.env_sock.close()
                del self.env_sock
            if hasattr(self, 'inf_sock'):
                self.inf_sock.close()
                del self.inf_sock
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