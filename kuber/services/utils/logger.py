"""
RL Logger Module - JSON-based logging for RL systems with multiple output handlers

Features:
- JSON-structured logging with nanosecond precision timestamps
- Multiple output handlers: Console, File, RabbitMQ
- Thread-safe operation for multi-threaded applications
- Flask integration support
- Comprehensive environment variable configuration
- Automatic reconnection and error handling

Usage:
    # Simple setup
    import rl_logger
    rl_logger.setup("snake_exp_001")
    
    import logging
    logging.info({"message": "Training started", "epoch": 1})
"""

import json
import logging
import sys
import os
import time
import uuid
import threading
import queue
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

# Optional dependencies
try:
    import pika
    from pika.exceptions import AMQPConnectionError, AMQPChannelError
    logging.getLogger("pika").propagate = False
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False


class LogMetadata:
    """Manages logging metadata (experiment name, run ID, container)"""
    
    def __init__(self, experiment_name: str = None, run_id: str = None, container: str = None):
        self.experiment_name = experiment_name or os.getenv('EXPERIMENT_NAME', 'default_exp')
        self.run_id = run_id or os.getenv('RUN_ID', self._generate_run_id())
        self.container = container or self._detect_container()
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID"""
        return f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    
    def _detect_container(self) -> str:
        """Auto-detect container name from various sources"""
        return (
            os.getenv('CONTAINER_NAME') or
            os.getenv('HOSTNAME', '').split('.')[0] or
            os.path.basename(sys.argv[0]).replace('.py', '') or
            'unknown'
        )
    
    def to_dict(self) -> Dict[str, str]:
        """Convert metadata to dictionary"""
        return {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "container": self.container
        }


class TimestampGenerator:
    """Generates high-precision timestamps"""
    
    @staticmethod
    def get_timestamp() -> str:
        """Get ISO timestamp with nanosecond precision"""
        now = datetime.now(timezone.utc)
        nanoseconds = time.time_ns() % 1_000_000_000
        timestamp_base = now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
        return f"{timestamp_base}{nanoseconds:03d}+00:00"


class LogFormatter:
    """Formats log records into JSON structure"""
    
    def __init__(self, metadata: LogMetadata):
        self.metadata = metadata
    
    def format(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Format log record into structured dictionary"""
        # Extract message data
        if isinstance(record.msg, dict):
            log_entry = record.msg.copy()
        elif hasattr(record, 'dict_data') and isinstance(record.dict_data, dict):
            log_entry = record.dict_data.copy()
        else:
            log_entry = {"message": str(record.msg) if record.msg else record.getMessage()}
        
        # Add metadata
        log_entry.update(self.metadata.to_dict())
        log_entry.update({
            "timestamp": TimestampGenerator.get_timestamp(),
            "level": record.levelname,
            "logger_name": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        })
        
        return log_entry


class BaseHandler(ABC):
    """Abstract base class for log handlers"""
    
    def __init__(self, metadata: LogMetadata, level: str = "INFO"):
        self.metadata = metadata
        self.formatter = LogFormatter(metadata)
        self.level = getattr(logging, level.upper())
        self.enabled = True
    
    @abstractmethod
    def emit(self, record: logging.LogRecord) -> bool:
        """Emit log record. Returns True if successful, False otherwise"""
        pass
    
    def should_emit(self, record: logging.LogRecord) -> bool:
        """Check if record should be emitted"""
        return self.enabled and record.levelno >= self.level
    
    def close(self):
        """Cleanup handler resources"""
        pass


class ConsoleHandler(BaseHandler):
    """Handler for console output"""
    
    def __init__(self, metadata: LogMetadata, level: str = "INFO", 
                 stream=None, use_tsv: bool = True):
        super().__init__(metadata, level)
        self.stream = stream or sys.stdout
        self.use_tsv = use_tsv
    
    def emit(self, record: logging.LogRecord) -> bool:
        """Emit to console"""
        if not self.should_emit(record):
            return True
        
        try:
            log_data = self.formatter.format(record)
            
            if self.use_tsv:
                # TSV format: timestamp + TAB + json
                timestamp = log_data.pop("timestamp")
                json_str = json.dumps(log_data, separators=(',', ':'))
                message = f"{timestamp}\t{json_str}"
            else:
                # Pure JSON format
                message = json.dumps(log_data, separators=(',', ':'))
            
            self.stream.write(message + '\n')
            self.stream.flush()
            return True
            
        except (BrokenPipeError, OSError):
            return False
        except Exception:
            return False


class FileHandler(BaseHandler):
    """Handler for file output"""
    
    def __init__(self, metadata: LogMetadata, filename: str, level: str = "INFO",
                 use_tsv: bool = True):
        super().__init__(metadata, level)
        self.filename = filename
        self.use_tsv = use_tsv
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Create directory if it doesn't exist"""
        log_dir = os.path.dirname(self.filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def emit(self, record: logging.LogRecord) -> bool:
        """Emit to file"""
        if not self.should_emit(record):
            return True
        
        try:
            log_data = self.formatter.format(record)
            
            if self.use_tsv:
                # TSV format: timestamp + TAB + json
                timestamp = log_data.pop("timestamp")
                json_str = json.dumps(log_data, separators=(',', ':'))
                message = f"{timestamp}\t{json_str}"
            else:
                # Pure JSON format
                message = json.dumps(log_data, separators=(',', ':'))
            
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
                f.flush()
            return True
            
        except (OSError, IOError):
            return False
        except Exception:
            return False


class RabbitMQHandler(BaseHandler):
    """Handler for RabbitMQ output"""
    
    def __init__(self, metadata: LogMetadata, level: str = "INFO"):
        if not RABBITMQ_AVAILABLE:
            raise ImportError("pika library not available. Install with: pip install pika")
        
        super().__init__(metadata, level)
        
        # RabbitMQ configuration from environment
        self.config = self._load_config()
        
        # Connection state
        self.connection = None
        self.channel = None
        self.is_connected = False
        self.retry_count = 0
        self.lock = threading.Lock()
        
        # Initialize connection
        self._connect()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load RabbitMQ configuration from environment variables"""
        return {
            'host': os.getenv('RABBITMQ_HOST', 'localhost'),
            'port': int(os.getenv('RABBITMQ_PORT', '5672')),
            'username': os.getenv('RABBITMQ_USERNAME', 'guest'),
            'password': os.getenv('RABBITMQ_PASSWORD', 'guest'),
            'vhost': os.getenv('RABBITMQ_VHOST', '/'),
            'exchange': os.getenv('RABBITMQ_EXCHANGE', 'logs'),
            'routing_key': os.getenv('RABBITMQ_ROUTING_KEY', 'rl_logs'),
            'exchange_type': os.getenv('RABBITMQ_EXCHANGE_TYPE', 'topic'),
            'durable': os.getenv('RABBITMQ_DURABLE', 'true').lower() == 'true',
            'connection_timeout': int(os.getenv('RABBITMQ_CONNECTION_TIMEOUT', '10')),
            'retry_delay': int(os.getenv('RABBITMQ_RETRY_DELAY', '5')),
            'max_retries': int(os.getenv('RABBITMQ_MAX_RETRIES', '3'))
        }
    
    def _connect(self):
        """Establish connection to RabbitMQ"""
        try:
            credentials = pika.PlainCredentials(self.config['username'], self.config['password'])
            parameters = pika.ConnectionParameters(
                host=self.config['host'],
                port=self.config['port'],
                virtual_host=self.config['vhost'],
                credentials=credentials,
                connection_attempts=3,
                retry_delay=2,
                socket_timeout=self.config['connection_timeout'],
                heartbeat=600,
                blocked_connection_timeout=300
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare exchange
            #self.channel.exchange_declare(
            #    exchange=self.config['exchange'],
            #    exchange_type=self.config['exchange_type'],
            #    durable=self.config['durable']
            #)
            
            self.is_connected = True
            self.retry_count = 0
            
        except Exception as e:
            self.is_connected = False
            if self.connection and not self.connection.is_closed:
                try:
                    self.connection.close()
                except:
                    pass
            self.connection = None
            self.channel = None
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to RabbitMQ"""
        if self.retry_count >= self.config['max_retries']:
            return False
        
        self.retry_count += 1
        time.sleep(self.config['retry_delay'])
        self._connect()
        return self.is_connected
    
    def emit(self, record: logging.LogRecord) -> bool:
        """Emit to RabbitMQ"""
        if not self.should_emit(record):
            return True
        
        with self.lock:
            try:
                log_data = self.formatter.format(record)
                json_message = json.dumps(log_data, separators=(',', ':'))
                
                if not self.is_connected:
                    if not self._reconnect():
                        return False
                
                # Publish message
                self.channel.basic_publish(
                    exchange=self.config['exchange'],
                    routing_key=self.config['routing_key'],
                    body=json_message,
                    properties=pika.BasicProperties(
                        delivery_mode=2 if self.config['durable'] else 1,
                        content_type='application/json',
                        timestamp=int(time.time())
                    )
                )
                return True
                
            except Exception:
                self.is_connected = False
                # Try to reconnect and publish again
                if self._reconnect():
                    try:
                        self.channel.basic_publish(
                            exchange=self.config['exchange'],
                            routing_key=self.config['routing_key'],
                            body=json_message,
                            properties=pika.BasicProperties(
                                delivery_mode=2 if self.config['durable'] else 1,
                                content_type='application/json',
                                timestamp=int(time.time())
                            )
                        )
                        return True
                    except Exception:
                        pass
                return False
    
    def close(self):
        """Close RabbitMQ connection"""
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.close()
            if self.connection and not self.connection.is_closed:
                self.connection.close()
        except Exception:
            pass
        finally:
            self.is_connected = False


class AsyncRabbitMQHandler(BaseHandler):
    """Asynchronous RabbitMQ handler using background thread"""
    
    def __init__(self, metadata: LogMetadata, level: str = "INFO", queue_size: int = 1000):
        super().__init__(metadata, level)
        
        # Message queue
        self.message_queue = queue.Queue(maxsize=queue_size)
        self.queue_size = queue_size
        
        # Background thread
        self.worker_thread = None
        self.stop_event = threading.Event()
        
        # Underlying RabbitMQ handler
        try:
            self.rabbitmq_handler = RabbitMQHandler(metadata, level)
            self._start_worker()
        except ImportError:
            raise
        except Exception:
            # RabbitMQ connection failed, but we still create the handler
            self.rabbitmq_handler = None
    
    def _start_worker(self):
        """Start background worker thread"""
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def _worker(self):
        """Background worker that processes the message queue"""
        while not self.stop_event.is_set():
            try:
                try:
                    record = self.message_queue.get(timeout=1.0)
                    if record is None:  # Shutdown signal
                        break
                    
                    if self.rabbitmq_handler:
                        self.rabbitmq_handler.emit(record)
                    self.message_queue.task_done()
                    
                except queue.Empty:
                    continue
                    
            except Exception:
                time.sleep(1)
    
    def emit(self, record: logging.LogRecord) -> bool:
        """Add record to queue for async processing"""
        if not self.should_emit(record):
            return True
        
        try:
            self.message_queue.put_nowait(record)
            return True
        except queue.Full:
            return False
    
    def close(self):
        """Shutdown async handler"""
        self.stop_event.set()
        
        try:
            self.message_queue.put_nowait(None)
        except queue.Full:
            pass
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        if self.rabbitmq_handler:
            self.rabbitmq_handler.close()


class LoggerConfig:
    """Configuration class for logger setup"""
    
    def __init__(self):
        self.experiment_name = None
        self.run_id = None
        self.container = None
        self.log_level = "INFO"
        self.enable_console = True
        self.enable_file = False
        self.enable_rabbitmq = False
        self.log_file = None
        self.rabbitmq_async = True
        self.flask_app = None
        self.thread_safe = None
        self.use_tsv_format = True
    
    @classmethod
    def from_env(cls) -> 'LoggerConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        config.experiment_name = os.getenv('EXPERIMENT_NAME')
        config.run_id = os.getenv('RUN_ID')
        config.container = os.getenv('CONTAINER_NAME')
        config.log_level = os.getenv('LOG_LEVEL', 'INFO')
        config.log_file = os.getenv('LOG_FILE')
        
        # Boolean environment variables
        config.enable_console = os.getenv('ENABLE_CONSOLE_LOGS', 'true').lower().strip('"\'') == 'true'
        config.enable_file = config.log_file is not None
        config.enable_rabbitmq = os.getenv('ENABLE_RABBITMQ', 'false').lower().strip('"\'') == 'true'
        config.rabbitmq_async = os.getenv('RABBITMQ_ASYNC', 'true').lower().strip('"\'') == 'true'
        config.use_tsv_format = os.getenv('USE_TSV_FORMAT', 'true').lower().strip('"\'') == 'true'
        
        return config


class RLLogger:
    """Main logger class that manages multiple handlers"""
    
    def __init__(self, config: LoggerConfig):
        self.config = config
        self.metadata = LogMetadata(
            experiment_name=config.experiment_name,
            run_id=config.run_id,
            container=config.container
        )
        self.handlers: List[BaseHandler] = []
        self.root_logger = logging.getLogger()
        self._setup_handlers()
        self._setup_logging()
    
    def _setup_handlers(self):
        """Setup all enabled handlers"""
        # Console handler
        if self.config.enable_console:
            handler = ConsoleHandler(
                self.metadata, 
                level=self.config.log_level,
                use_tsv=self.config.use_tsv_format
            )
            self.handlers.append(handler)
        
        # File handler
        if self.config.enable_file and self.config.log_file:
            handler = FileHandler(
                self.metadata, 
                filename=self.config.log_file,
                level=self.config.log_level,
                use_tsv=self.config.use_tsv_format
            )
            self.handlers.append(handler)
        
        # RabbitMQ handler
        if self.config.enable_rabbitmq:
            try:
                if self.config.rabbitmq_async:
                    handler = AsyncRabbitMQHandler(self.metadata, level=self.config.log_level)
                else:
                    handler = RabbitMQHandler(self.metadata, level=self.config.log_level)
                self.handlers.append(handler)
            except ImportError:
                sys.stdout.write("⚠️  RabbitMQ handler requested but pika not installed\n")
            except Exception as e:
                sys.stdout.write(f"⚠️  Failed to setup RabbitMQ handler: {e}\n")
    
    def _setup_logging(self):
        """Setup Python logging system"""
        # Clear existing handlers
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)
        
        # Set log level
        self.root_logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Add our custom handler
        custom_handler = MultiHandler(self.handlers)
        self.root_logger.addHandler(custom_handler)
        self.root_logger.propagate = False
        
        # Handle Flask integration
        if self.config.flask_app:
            self._setup_flask_logging()
        
        self._print_status()
    
    def _setup_flask_logging(self):
        """Setup Flask-specific logging"""
        flask_app = self.config.flask_app
        
        # Configure Flask's logger
        flask_app.logger.handlers.clear()
        flask_app.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        flask_app.logger.addHandler(MultiHandler(self.handlers))
        flask_app.logger.propagate = False
        
        # Configure werkzeug logger
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.handlers.clear()
        werkzeug_logger.setLevel(logging.WARNING)
        werkzeug_logger.addHandler(MultiHandler(self.handlers))
        werkzeug_logger.propagate = False
    
    def _print_status(self):
        """Print setup status message"""
        handlers_info = []
        if self.config.enable_console:
            handlers_info.append("Console")
        if self.config.enable_file:
            handlers_info.append(f"File({self.config.log_file})")
        if self.config.enable_rabbitmq:
            mode = "Async" if self.config.rabbitmq_async else "Sync"
            handlers_info.append(f"RabbitMQ({mode})")
        
        status = f"✅ RL Logger initialized (level:{self.config.log_level})- {self.metadata.experiment_name}:{self.metadata.run_id}@{self.metadata.container} "
        if handlers_info:
            status += f" -> {', '.join(handlers_info)}"
        
        sys.stdout.write(status + '\n')
        sys.stdout.flush()
    
    def close(self):
        """Close all handlers"""
        for handler in self.handlers:
            handler.close()


class MultiHandler(logging.Handler):
    """Logging handler that dispatches to multiple custom handlers"""
    
    def __init__(self, handlers: List[BaseHandler]):
        super().__init__()
        self.handlers = handlers
    
    def emit(self, record):
        """Emit to all handlers"""
        for handler in self.handlers:
            try:
                success = handler.emit(record)
                if not success and isinstance(handler, (RabbitMQHandler, AsyncRabbitMQHandler)):
                    # Log RabbitMQ failures to stderr
                    log_data = handler.formatter.format(record)
                    json_str = json.dumps(log_data, separators=(',', ':'))
                    sys.stdout.write(f"RABBITMQ_FAILED: {json_str}\n")
                    sys.stdout.flush()
            except Exception:
                self.handleError(record)
    
    def close(self):
        """Close all handlers"""
        for handler in self.handlers:
            handler.close()
        super().close()


# Global logger instance
_global_logger: Optional[RLLogger] = None


def setup(experiment_name: str = None, 
          run_id: str = None,
          container: str = None,
          log_file: str = None,
          log_level: str = None,
          enable_console: bool = None,
          enable_rabbitmq: bool = None,
          rabbitmq_async: bool = None,
          flask_app=None,
          use_env: bool = True) -> RLLogger:
    """
    Setup RL Logger with specified configuration
    
    Args:
        experiment_name: Name of the experiment
        run_id: Unique run identifier
        container: Container/service name
        log_file: Path to log file (enables file logging)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Enable console output
        enable_rabbitmq: Enable RabbitMQ output
        rabbitmq_async: Use async RabbitMQ handler
        flask_app: Flask app instance for integration
        use_env: Whether to read from environment variables
    
    Returns:
        RLLogger instance
    """
    global _global_logger
    
    # Start with environment config if requested
    if use_env:
        config = LoggerConfig.from_env()
    else:
        config = LoggerConfig()
    
    # Override with explicit parameters
    if experiment_name is not None:
        config.experiment_name = experiment_name
    if run_id is not None:
        config.run_id = run_id
    if container is not None:
        config.container = container
    if log_file is not None:
        config.log_file = log_file
        config.enable_file = True
    if log_level is not None:
        config.log_level = log_level
    if enable_console is not None:
        config.enable_console = enable_console
    if enable_rabbitmq is not None:
        config.enable_rabbitmq = enable_rabbitmq
    if rabbitmq_async is not None:
        config.rabbitmq_async = rabbitmq_async
    if flask_app is not None:
        config.flask_app = flask_app
        config.thread_safe = True
    
    # Close existing logger
    if _global_logger:
        _global_logger.close()
    
    # Create new logger
    _global_logger = RLLogger(config)
    return _global_logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance"""
    if _global_logger is None:
        setup()  # Setup with defaults
    return logging.getLogger(name)


# Convenience functions
def info(data: Dict[str, Any]):
    """Log INFO level"""
    logging.info(data)

def debug(data: Dict[str, Any]):
    """Log DEBUG level"""
    logging.debug(data)

def warning(data: Dict[str, Any]):
    """Log WARNING level"""
    logging.warning(data)

def error(data: Dict[str, Any]):
    """Log ERROR level"""
    logging.error(data)

def critical(data: Dict[str, Any]):
    """Log CRITICAL level"""
    logging.critical(data)


def test_logging():
    """Test function to verify logging works"""
    sys.stdout.write("=== Testing RL Logger ===\n")
    sys.stdout.flush()
    
    # Test all levels
    logging.critical({"message": "CRITICAL test", "level": "critical"})
    logging.error({"message": "ERROR test", "level": "error"})
    logging.warning({"message": "WARNING test", "level": "warning"})
    logging.info({"message": "INFO test", "level": "info"})
    logging.debug({"message": "DEBUG test", "level": "debug"})
    
    # Test plain strings
    logging.info("Plain string test")
    
    sys.stdout.write("=== End Test ===\n")
    sys.stdout.flush()


# Backward compatibility aliases
setup_as_default = setup  # For backward compatibility with your existing code


if __name__ == "__main__":
    # Example usage
    setup(
        experiment_name="snake_exp_001",
        log_file="logs/test.log",
        enable_rabbitmq=True,
        rabbitmq_async=True
    )
    
    # Run test
    test_logging()
    
    # Example usage
    import logging
    
    logging.info({"message": "Episode completed", "reward": 133.5})
    logging.error({"component": "training", "error_code": "CUDA_OOM"})
    logging.debug({"component": "env", "action": "forward"})
    
    # Plain strings also work
    logging.info("This is a plain message")
    logging.warning("Something went wrong")
