"""
RL Logger Module - JSON-based logging for RL systems
To use as default logger across entire project:
    # At the very start of your main.py or __init__.py:
    import rl_logger
    rl_logger.setup_as_default("snake_exp_001", log_file="logs/app.log")
    
    # Then anywhere in your code:
    import logging
    logging.info({"message": "This will be JSON!"})
"""
import json
import logging
import sys
import os
import time
import uuid
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional

class DictJSONFormatter(logging.Formatter):
    """Formatter that takes dict input and adds timestamp + metadata"""
    
    def __init__(self, experiment_name: str, run_id: str):
        super().__init__()
        self.experiment_name = experiment_name
        self.run_id = run_id
    
    def format(self, record):
        # Get timestamp with nanosecond precision
        now = datetime.now(timezone.utc)
        # Get nanoseconds from time.time_ns()
        nanoseconds = time.time_ns() % 1_000_000_000  # Get nanosecond part
        # Format with microseconds and add nanoseconds
        timestamp_base = now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]  # Remove last 3 digits of microseconds
        timestamp = f"{timestamp_base}{nanoseconds:03d}+00:00"  # Add nanoseconds and timezone
        
        # Get the dictionary from the log record
        if hasattr(record, 'dict_data') and isinstance(record.dict_data, dict):
            # Start with the user's dictionary
            log_entry = record.dict_data.copy()
            
            # Add our metadata (these will override if user provided them)
            log_entry.update({
                "experiment_name": self.experiment_name,
                "run_id": self.run_id,
                "level": record.levelname
            })
        else:
            # Fallback for non-dict messages
            log_entry = {
                "experiment_name": self.experiment_name,
                "run_id": self.run_id,
                "level": record.levelname,
                "message": record.getMessage()
            }
        
        # Output: timestamp + space + json
        json_str = json.dumps(log_entry, separators=(',', ':'))
        return f"{timestamp} {json_str}"

class DictJSONLogger:
    """Logger that accepts dictionaries and adds metadata automatically"""
    
    def __init__(self, 
                 name: str = "rl_logger",
                 experiment_name: str = None,
                 run_id: str = None,
                 log_file: str = None,
                 log_level: str = "INFO"):
        
        # Get from env vars or use defaults
        self.experiment_name = experiment_name or os.getenv('EXPERIMENT_NAME', 'default_exp')
        self.run_id = run_id or os.getenv('RUN_ID', f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}")
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = DictJSONFormatter(self.experiment_name, self.run_id)
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.propagate = False
    
    def _log_dict(self, level: str, data: Dict[str, Any]):
        """Internal method to log dictionary with specific level"""
        # Create a custom log record with the dict data
        log_method = getattr(self.logger, level.lower())
        
        # Monkey patch to inject our dict
        original_makeRecord = self.logger.makeRecord
        def makeRecord(*args, **kwargs):
            record = original_makeRecord(*args, **kwargs)
            record.dict_data = data
            return record
        
        self.logger.makeRecord = makeRecord
        log_method("")  # Empty message since we use dict_data
        self.logger.makeRecord = original_makeRecord  # Restore
    
    def debug(self, data: Dict[str, Any]):
        """Log DEBUG level with dictionary data"""
        self._log_dict("DEBUG", data)
    
    def info(self, data: Dict[str, Any]):
        """Log INFO level with dictionary data"""
        self._log_dict("INFO", data)
    
    def warning(self, data: Dict[str, Any]):
        """Log WARNING level with dictionary data"""
        self._log_dict("WARNING", data)
    
    def error(self, data: Dict[str, Any]):
        """Log ERROR level with dictionary data"""
        self._log_dict("ERROR", data)
    
    def critical(self, data: Dict[str, Any]):
        """Log CRITICAL level with dictionary data"""
        self._log_dict("CRITICAL", data)

# Factory function to create logger with specific experiment/run
def create_logger(experiment_name: str, 
                  run_id: Optional[str] = None,
                  name: Optional[str] = None,
                  log_file: Optional[str] = None,
                  log_level: str = "INFO",
                  enable_stdout: bool = True) -> DictJSONLogger:
    """
    Create a logger with specific experiment name and run ID
    
    Args:
        experiment_name: Name of the experiment (e.g., "snake_exp_001")
        run_id: Optional run ID. If None, auto-generated
        name: Optional logger name. If None, uses experiment_name
        log_file: Optional file path for logging. If None, only console output
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_stdout: Whether to output to console (default: True)
    
    Returns:
        DictJSONLogger instance
    """
    logger_name = name or f"rl_logger_{experiment_name}"
    return DictJSONLogger(
        name=logger_name,
        experiment_name=experiment_name,
        run_id=run_id,
        log_file=log_file,
        log_level=log_level
    )

# Global configuration for default logging
_default_experiment_name = None
_default_run_id = None
_is_default_setup = False

class DefaultLogAdapter(logging.LoggerAdapter):
    """Adapter that converts regular logging calls to JSON format"""
    
    def process(self, msg, kwargs):
        # If msg is already a dict, use it directly
        if isinstance(msg, dict):
            return msg, kwargs
        
        # If msg is a string, wrap it in message field
        return {"message": str(msg)}, kwargs

class DictJSONHandler(logging.Handler):
    """Custom handler that processes both dict and string messages"""
    
    def __init__(self, stream=None, filename=None, thread_safe=False):
        super().__init__()
        self.experiment_name = _default_experiment_name or "default_exp"
        self.run_id = _default_run_id or f"run_{int(time.time())}"
        
        # Thread lock only if requested (for Flask/multi-threaded apps)
        self._lock = threading.Lock() if thread_safe else None
        
        # Setup output streams - FIXED LOGIC
        self.enable_console = stream is not None
        self.enable_file = filename is not None
        self.log_filename = filename
        
        if filename:
            # Create directory if needed
            log_dir = os.path.dirname(filename)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
    
    def emit(self, record):
        try:
            # Use lock only if thread_safe is enabled
            if self._lock:
                with self._lock:
                    self._do_emit(record)
            else:
                self._do_emit(record)
                        
        except Exception:
            # Don't let logging errors crash the app
            self.handleError(record)
    
    def _do_emit(self, record):
        """Internal emit method without locking"""
        # Get timestamp with nanosecond precision
        now = datetime.now(timezone.utc)
        # Get nanoseconds from time.time_ns()
        nanoseconds = time.time_ns() % 1_000_000_000  # Get nanosecond part
        # Format with microseconds and add nanoseconds
        timestamp_base = now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]  # Remove last 3 digits of microseconds
        timestamp = f"{timestamp_base}{nanoseconds:03d}+00:00"  # Add nanoseconds and timezone
        
        # Process the message - FIXED DICT HANDLING
        if isinstance(record.msg, dict):
            log_entry = record.msg.copy()
        elif hasattr(record, 'dict_data') and isinstance(record.dict_data, dict):
            log_entry = record.dict_data.copy()
        else:
            log_entry = {"message": str(record.msg) if record.msg else record.getMessage()}
        
        # Add metadata
        log_entry.update({
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "level": record.levelname
        })
        
        # Create final log line
        json_str = json.dumps(log_entry, separators=(',', ':'))
        final_message = f"{timestamp} {json_str}"
        
        # Output to console - FIXED FOR DOCKER
        if self.enable_console:
            try:
                sys.stdout.write(final_message + '\n')
                sys.stdout.flush()
            except (BrokenPipeError, OSError):
                pass
        
        # Output to file
        if self.enable_file and self.log_filename:
            try:
                with open(self.log_filename, 'a', encoding='utf-8') as f:
                    f.write(final_message + '\n')
                    f.flush()
            except (OSError, IOError):
                pass

def setup_as_default(experiment_name: str = None, 
                     run_id: Optional[str] = None,
                     log_file: Optional[str] = None,
                     log_level: str = "INFO",
                     enable_stdout: bool = None,
                     flask_app=None,
                     thread_safe: bool = None):
    """
    Setup this logger as the default logging system for the entire application
    
    Args:
        experiment_name: Name of the experiment (defaults to env var EXPERIMENT_NAME or 'default_exp')
        run_id: Optional run ID. If None, uses env var RUN_ID or auto-generated
        log_file: Optional file path for logging (can use env var LOG_FILE)
        log_level: Log level (defaults to env var LOG_LEVEL or 'INFO')
        enable_stdout: Whether to output to console (defaults to env var ENABLE_CONSOLE_LOGS or True)
        flask_app: Flask app instance for proper Flask integration
        thread_safe: Enable thread safety (auto-enabled for Flask, disabled otherwise)
    
    Usage:
        # Single-threaded services (training, inference):
        rl_logger.setup_as_default()  # No threading overhead
        
        # Flask/multi-threaded services:
        rl_logger.setup_as_default(flask_app=app)  # Thread-safe enabled
        
        # Force thread safety:
        rl_logger.setup_as_default(thread_safe=True)
    """
    global _default_experiment_name, _default_run_id, _is_default_setup
    
    # Get values from env vars with fallbacks
    _default_experiment_name = experiment_name or os.getenv('EXPERIMENT_NAME', 'default_exp')
    _default_run_id = run_id or os.getenv('RUN_ID', f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}")
    
    # Get other settings from env vars if not provided
    final_log_file = log_file or os.getenv('LOG_FILE')
    final_log_level = os.getenv('LOG_LEVEL', log_level)
    
    # Handle enable_stdout from env var - FIX: strip quotes
    if enable_stdout is None:
        env_val = os.getenv('ENABLE_CONSOLE_LOGS', 'true')
        # Strip quotes if present
        env_val = env_val.strip().strip('"').strip("'").lower()
        enable_stdout = env_val == 'true'
    
    # Auto-enable thread safety for Flask, otherwise disable by default
    if thread_safe is None:
        thread_safe = flask_app is not None
    
    _is_default_setup = True
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Remove all existing handlers to avoid conflicts
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    root_logger.setLevel(getattr(logging, final_log_level.upper()))
    
    # Add our custom handler with thread safety control
    handler = DictJSONHandler(
        stream=sys.stdout if enable_stdout else None, 
        filename=final_log_file,
        thread_safe=thread_safe
    )
    root_logger.addHandler(handler)
    
    # Prevent propagation to avoid duplicates
    root_logger.propagate = False
    
    # Handle Flask app logging specially (after root logger setup)
    if flask_app is not None:
        # Configure Flask's logger to use the same handler
        flask_app.logger.handlers.clear()
        flask_app.logger.setLevel(getattr(logging, final_log_level.upper()))
        flask_app.logger.addHandler(handler)  # Use same handler
        flask_app.logger.propagate = False
        
        # Also configure werkzeug logger (Flask's HTTP server)
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.handlers.clear()
        werkzeug_logger.setLevel(logging.WARNING)
        werkzeug_logger.addHandler(handler)  # Use same handler
        werkzeug_logger.propagate = False
    
    status_msg = f"✅ Default logging setup - Experiment: {_default_experiment_name}, Run: {_default_run_id}"
    if final_log_file and enable_stdout:
        status_msg += f" (Console + File: {final_log_file})"
    elif final_log_file:
        status_msg += f" (File only: {final_log_file})"
    else:
        status_msg += " (Console only)"
    
    if flask_app:
        status_msg += " [Flask integrated]"
    elif thread_safe:
        status_msg += " [Thread-safe]"
    else:
        status_msg += " [Single-threaded]"
    
    # Use direct write for setup message
    sys.stdout.write(status_msg + '\n')
    sys.stdout.flush()
    
    # Create debug messages using our JSON format
    def debug_log(message, **data):
        now = datetime.now(timezone.utc)
        # Get nanoseconds from time.time_ns()
        nanoseconds = time.time_ns() % 1_000_000_000  # Get nanosecond part
        # Format with microseconds and add nanoseconds
        timestamp_base = now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]  # Remove last 3 digits of microseconds
        timestamp = f"{timestamp_base}{nanoseconds:03d}+00:00"  # Add nanoseconds and timezone
        
        log_entry = {
            "experiment_name": _default_experiment_name,
            "run_id": _default_run_id,
            "level": "DEBUG",
            "message": message,
            **data
        }
        json_str = json.dumps(log_entry, separators=(',', ':'))
        sys.stdout.write(f"{timestamp} {json_str}\n")
        sys.stdout.flush()
    
    # DEBUG: Check logger configuration using JSON format
    debug_log("Logger configuration check", 
              root_logger_level=root_logger.level,
              info_level=logging.INFO,
              effective_level=root_logger.getEffectiveLevel(),
              handler_count=len(root_logger.handlers),
              final_log_level=final_log_level,
              parsed_log_level=getattr(logging, final_log_level.upper()),
              enable_console_logs_env=os.getenv('ENABLE_CONSOLE_LOGS'),
              enable_stdout_processed=enable_stdout)
    
    for i, h in enumerate(root_logger.handlers):
        debug_log(f"Handler {i} configuration",
                  handler_type=type(h).__name__,
                  enable_console=getattr(h, 'enable_console', None),
                  enable_file=getattr(h, 'enable_file', None),
                  log_filename=getattr(h, 'log_filename', None))
    
    # IMMEDIATE TEST - Force direct call to handler
    debug_log("Testing handler directly")
    
    # Test with INFO level (should work)
    try:
        test_record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg={"message": "Direct handler test", "test": True}, 
            args=(), exc_info=None
        )
        handler._do_emit(test_record)
        debug_log("Direct handler call completed successfully")
    except Exception as e:
        debug_log("Direct handler call failed", error=str(e))
    
    # Test normal logging
    debug_log("Testing normal logging calls")
    
    root_logger.error({"message": "ERROR level test", "should": "appear"})
    root_logger.warning({"message": "WARNING level test", "should": "appear"})
    root_logger.info({"message": "INFO level test", "should": "appear"})
    
    debug_log("Normal logging test completed")

def get_logger(name: Optional[str] = None):
    """
    Get a logger instance that works with the default setup
    
    Args:
        name: Optional logger name
        
    Returns:
        Logger that outputs JSON format
    """
    if not _is_default_setup:
        raise RuntimeError("Must call setup_as_default() first!")
    
    logger = logging.getLogger(name)
    return logger

# Test function for debugging
def test_logging():
    """Test function to verify logging works"""
    # Note: logging is already imported at module level
    
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

# Convenience functions (FIXED)
def get_default_logger():
    """Get default logger"""
    if not _is_default_setup:
        setup_as_default()
    return logging.getLogger()

def info(data: Dict[str, Any]):
    """Log INFO level using default logger"""
    logging.info(data)

def debug(data: Dict[str, Any]):
    """Log DEBUG level using default logger"""
    logging.debug(data)

def warning(data: Dict[str, Any]):
    """Log WARNING level using default logger"""
    logging.warning(data)

def error(data: Dict[str, Any]):
    """Log ERROR level using default logger"""
    logging.error(data)

def critical(data: Dict[str, Any]):
    """Log CRITICAL level using default logger"""
    logging.critical(data)

# Example usage
if __name__ == "__main__":
    # Setup as default logger
    setup_as_default("snake_exp_001", log_file="logs/test.log")
    
    # Run test
    test_logging()
    
    # Now use regular Python logging anywhere in your code
    # Note: logging is already imported at module level
    
    # These all work and output JSON:
    logging.info({"message": "Episode completed", "reward": 133.5})
    logging.error({"component": "training", "error_code": "CUDA_OOM"})
    logging.debug({"component": "env", "action": "forward"})
    
    # Plain strings also work:
    logging.info("This is a plain message")
    logging.warning("Something went wrong")
    
    # Different loggers all use the same JSON format:
    training_logger = logging.getLogger("training")
    inference_logger = logging.getLogger("inference")
    
    training_logger.info({"component": "training", "message": "Batch processed"})
    inference_logger.error({"component": "inference", "message": "Model failed"})
