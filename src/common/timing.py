"""Timing utilities for performance monitoring."""

import os
import time
from datetime import datetime
from functools import wraps
from typing import Callable, Any


def get_timestamp() -> str:
    """Get current timestamp in format [HH:MM:SS.mmm]."""
    return datetime.now().strftime("[%H:%M:%S.%f")[:-3] + "]"


def timed(func: Callable) -> Callable:
    """Decorator to measure and log execution time if DEBUG_TIMING is enabled.

    Usage:
        @timed
        def my_function():
            ...

    Output (if DEBUG_TIMING=1):
        [12:34:56.789] [TIMING] my_function took 1.234s
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        debug = os.environ.get("DEBUG_TIMING", "").lower() in {"1", "true", "on"}
        if not debug:
            return func(*args, **kwargs)

        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        # Get function name with class if it's a method
        func_name = func.__name__
        if args and hasattr(args[0].__class__, func.__name__):
            func_name = f"{args[0].__class__.__name__}.{func.__name__}"

        from .log_utils import log_debug
        log_debug(f"{func_name} took {elapsed:.3f}s", "TIMING")
        return result

    return wrapper
