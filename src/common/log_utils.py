"""Centralized logging utilities with tqdm support and timestamps."""
from __future__ import annotations

import os
import sys
from datetime import datetime

# Check if tqdm is available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ANSI color codes
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_GRAY = "\033[90m"

# Log level configuration
LOG_LEVEL_DEBUG = 0
LOG_LEVEL_INFO = 1
LOG_LEVEL_WARNING = 2
LOG_LEVEL_ERROR = 3

# Environment variable for log level (default INFO)
_LOG_LEVEL = LOG_LEVEL_INFO
_env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
if _env_level == "DEBUG":
    _LOG_LEVEL = LOG_LEVEL_DEBUG
elif _env_level == "WARNING":
    _LOG_LEVEL = LOG_LEVEL_WARNING
elif _env_level == "ERROR":
    _LOG_LEVEL = LOG_LEVEL_ERROR

# Debug flags from environment
DEBUG_WORKSPACE = os.environ.get("DEBUG_WORKSPACE", "").lower() in {"1", "true", "on"}
DEBUG_HANDLER = os.environ.get("DEBUG_HANDLER", "").lower() in {"1", "true", "on"}
DEBUG_VIEWER = os.environ.get("DEBUG_VIEWER", "").lower() in {"1", "true", "on"}
DEBUG_TIMING = os.environ.get("DEBUG_TIMING", "").lower() in {"1", "true", "on"}


def get_timestamp() -> str:
    """Get current timestamp in [HH:MM:SS.mmm] format."""
    return datetime.now().strftime("[%H:%M:%S.%f]")[:-3]


def _format_message(message: str, prefix: str = "", level: str = "", color: str = "") -> str:
    """Format a log message with timestamp, optional prefix, and color."""
    timestamp = get_timestamp()
    parts = [timestamp]
    if level:
        parts.append(f"[{level}]")
    if prefix:
        parts.append(f"[{prefix}]")
    parts.append(message)
    formatted = " ".join(parts)

    if color and sys.stderr.isatty():  # Only use colors if output is a terminal
        return f"{color}{formatted}{COLOR_RESET}"
    return formatted


def _write(message: str, file=None) -> None:
    """Write a message using tqdm.write if available, otherwise print."""
    if file is None:
        file = sys.stdout

    if TQDM_AVAILABLE:
        tqdm.write(message, file=file)
    else:
        print(message, file=file, flush=True)

def log_perf(message: str, prefix: str = "PERF") -> None:
    if not os.environ.get("PERF_LOG", "").lower() not in {"", "0", "false", "off"}:
        return
    log_debug(message, prefix)

def log_debug(message: str, prefix: str = "") -> None:
    """Log a debug message (only if DEBUG level enabled)."""
    if _LOG_LEVEL <= LOG_LEVEL_DEBUG:
        formatted = _format_message(message, prefix, "DEBUG", COLOR_GRAY)
        _write(formatted, sys.stderr)


def log_info(message: str, prefix: str = "") -> None:
    """Log an info message."""
    if _LOG_LEVEL <= LOG_LEVEL_INFO:
        formatted = _format_message(message, prefix)
        _write(formatted)


def log_warning(message: str, prefix: str = "") -> None:
    """Log a warning message."""
    if _LOG_LEVEL <= LOG_LEVEL_WARNING:
        formatted = _format_message(message, prefix, "WARNING", COLOR_YELLOW)
        _write(formatted, sys.stderr)


def log_error(message: str, prefix: str = "") -> None:
    """Log an error message."""
    if _LOG_LEVEL <= LOG_LEVEL_ERROR:
        formatted = _format_message(message, prefix, "ERROR", COLOR_RED)
        _write(formatted, sys.stderr)


def is_debug_enabled(category: str = "") -> bool:
    """Check if debug is enabled globally or for a specific category."""
    if _LOG_LEVEL <= LOG_LEVEL_DEBUG:
        return True

    if category == "workspace":
        return DEBUG_WORKSPACE
    elif category == "handler":
        return DEBUG_HANDLER
    elif category == "viewer":
        return DEBUG_VIEWER
    elif category == "timing":
        return DEBUG_TIMING

    return False


