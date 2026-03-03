"""
Centralized logging configuration for the pipeline.

Replaces all print() calls with structured logging.
Outputs to both console (with color) and file.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path


_CONFIGURED = False


def setup_logger(
    name: str = "rag_pipeline",
    log_file: str = "output/pipeline.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return the pipeline logger.
    
    Args:
        name: Logger name.
        log_file: Path to the log file.
        level: Logging level.
        
    Returns:
        Configured logger instance.
    """
    global _CONFIGURED
    
    logger = logging.getLogger(name)
    
    if _CONFIGURED:
        return logger
    
    logger.setLevel(level)
    logger.propagate = False

    # Console handler with concise format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler with detailed format
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s │ %(name)s │ %(levelname)-7s │ %(filename)s:%(lineno)d │ %(message)s"
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    except (OSError, PermissionError):
        logger.warning(f"Could not create log file at {log_file}, logging to console only.")

    _CONFIGURED = True
    return logger


def get_logger(name: str = "rag_pipeline") -> logging.Logger:
    """
    Get the pipeline logger. Initializes with defaults if not yet configured.
    
    Args:
        name: Logger name (use dotted notation for sub-loggers).
        
    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
