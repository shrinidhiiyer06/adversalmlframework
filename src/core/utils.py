"""
Utility functions for reproducibility, logging, and benchmarking.

Provides core utilities used across the project for consistent behavior,
deterministic results, and performance monitoring.
"""

import numpy as np
import torch
import random
import logging
import time
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Ensure deterministic results across all libraries.
    
    Sets random seeds for Python's random module, NumPy, PyTorch (CPU and GPU),
    and configures CUDNN for deterministic behavior.
    
    Args:
        seed: Random seed value (default: 42)
        
    Note:
        Using deterministic CUDNN operations may slightly reduce performance
        but ensures reproducibility across runs.
        
    Example:
        >>> set_seed(42)
        >>> np.random.rand(5)  # Will always produce same values
        >>> set_seed(42)
        >>> np.random.rand(5)  # Same values as above
    """
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            # Ensure CUDNN is deterministic (slightly slower but reproducible)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            logger.debug(f"Set seed {seed} for CPU and {torch.cuda.device_count()} GPU(s)")
        else:
            logger.debug(f"Set seed {seed} (CPU only, CUDA not available)")
            
    except Exception as e:
        logger.warning(f"Failed to set all seeds: {e}")


def setup_logging(level: int = logging.INFO, format_style: str = 'detailed') -> None:
    """
    Configure professional logging with customizable format.
    
    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_style: Format style - 'detailed' or 'simple'
        
    Example:
        >>> setup_logging(level=logging.DEBUG)
        >>> logger.info("This is an info message")
    """
    if format_style == 'detailed':
        format_str = '%(asctime)s [%(levelname)-8s] %(name)-25s: %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
    else:
        format_str = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        datefmt = '%H:%M:%S'
    
    # Remove existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt=datefmt,
        force=True  # Override any existing configuration
    )
    
    # Suppress verbose libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logger.info(f"Logging configured: level={logging.getLevelName(level)}, style={format_style}")


def benchmark_function(func: Callable) -> Callable:
    """
    Decorator to measure and log execution time of a function.
    
    Unlike the old buggy decorator, this one:
    1. Only logs the time, doesn't change return values
    2. Preserves the original function signature
    3. Uses proper function wrapping
    
    Args:
        func: Function to benchmark
        
    Returns:
        Wrapped function that logs execution time
        
    Example:
        >>> @benchmark_function
        ... def slow_operation(n):
        ...     time.sleep(n)
        ...     return n * 2
        >>> result = slow_operation(0.1)  # Logs execution time
        >>> print(result)  # Still returns 2 * n
        
    Note:
        This decorator DOES NOT change the return value of the function.
        It only adds timing information to the logs.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_logger = logging.getLogger(func.__module__)
        start = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            
            func_logger.debug(
                f"{func.__name__} execution time: {elapsed_ms:.2f}ms"
            )
            
            return result  # Return original result unchanged
            
        except Exception as e:
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            func_logger.error(
                f"{func.__name__} failed after {elapsed_ms:.2f}ms: {str(e)}"
            )
            raise  # Re-raise the original exception
    
    return wrapper


def measure_latency(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """
    Measure execution time of a function call and return both result and latency.
    
    This is a utility function (not a decorator) that you call explicitly
    when you want both the result AND the timing information.
    
    Args:
        func: Function to call
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Tuple of (result, latency_ms)
        
    Example:
        >>> def add(a, b):
        ...     return a + b
        >>> result, latency = measure_latency(add, 2, 3)
        >>> print(f"Result: {result}, took {latency:.2f}ms")
        Result: 5, took 0.05ms
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    
    return result, latency_ms


def validate_numpy_array(
    arr: Any,
    name: str = "array",
    expected_shape: tuple = None,
    expected_dtype: type = None,
    check_finite: bool = True
) -> None:
    """
    Validate a numpy array meets requirements.
    
    Args:
        arr: Array to validate
        name: Name for error messages
        expected_shape: Expected shape tuple (None to skip check)
        expected_dtype: Expected dtype (None to skip check)
        check_finite: Whether to check for NaN/inf values
        
    Raises:
        TypeError: If arr is not a numpy array
        ValueError: If validation fails
        
    Example:
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> validate_numpy_array(arr, "features", expected_shape=(None, 2))
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be numpy array, got {type(arr)}")
    
    if expected_shape is not None:
        if len(arr.shape) != len(expected_shape):
            raise ValueError(
                f"{name} has {len(arr.shape)}D shape {arr.shape}, "
                f"expected {len(expected_shape)}D"
            )
        
        for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{name} dimension {i} is {actual}, expected {expected}"
                )
    
    if expected_dtype is not None and arr.dtype != expected_dtype:
        raise TypeError(
            f"{name} has dtype {arr.dtype}, expected {expected_dtype}"
        )
    
    if check_finite and not np.isfinite(arr).all():
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        raise ValueError(
            f"{name} contains {n_nan} NaN and {n_inf} inf values"
        )


def create_directory_safe(path: str) -> None:
    """
    Create directory if it doesn't exist, with error handling.
    
    Args:
        path: Directory path to create
        
    Raises:
        IOError: If directory cannot be created
    """
    import os
    try:
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise IOError(f"Cannot create directory {path}: {e}") from e


def format_bytes(num_bytes: float) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration into human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)}h {int(minutes)}m"


class ProgressTracker:
    """
    Simple progress tracker for long-running operations.
    
    Example:
        >>> tracker = ProgressTracker(100, "Processing samples")
        >>> for i in range(100):
        ...     # Do work
        ...     tracker.update()
        >>> tracker.finish()
    """
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.log_interval = 5.0  # Log every 5 seconds
        
    def update(self, amount: int = 1) -> None:
        """Update progress by amount."""
        self.current += amount
        
        # Log progress periodically
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self._log_progress()
            self.last_log_time = current_time
    
    def _log_progress(self) -> None:
        """Log current progress."""
        pct = (self.current / self.total) * 100 if self.total > 0 else 0
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
            logger.info(
                f"{self.description}: {self.current}/{self.total} "
                f"({pct:.1f}%) - ETA: {format_duration(eta)}"
            )
        else:
            logger.info(f"{self.description}: {self.current}/{self.total} ({pct:.1f}%)")
    
    def finish(self) -> None:
        """Log final progress."""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        logger.info(
            f"{self.description}: Complete! "
            f"Processed {self.current} items in {format_duration(elapsed)} "
            f"({rate:.1f} items/sec)"
        )
