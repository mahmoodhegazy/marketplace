"""
Logging Configuration for Freak AI
===================================

Provides structured logging with file rotation and rich console output.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    enable_console: bool = True,
    serialize: bool = False,
) -> None:
    """
    Configure the application logger.
    
    Parameters
    ----------
    log_level : str
        Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str, optional
        Path to log file. If provided, enables file logging.
    rotation : str
        When to rotate the log file (e.g., "10 MB", "1 day").
    retention : str
        How long to keep old log files.
    enable_console : bool
        Whether to output logs to console.
    serialize : bool
        Whether to output logs in JSON format.
    """
    # Remove default handler
    logger.remove()
    
    # Console format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # File format (more detailed)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    # Add console handler
    if enable_console:
        logger.add(
            sys.stderr,
            format=console_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
    
    # Add file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=file_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="gz",
            serialize=serialize,
            backtrace=True,
            diagnose=True,
        )


def get_logger(name: str = "freak_ai"):
    """
    Get a logger instance with the given name.
    
    Parameters
    ----------
    name : str
        Logger name (typically module name).
    
    Returns
    -------
    logger
        Configured logger instance.
    """
    return logger.bind(name=name)


class LoggerContext:
    """
    Context manager for structured logging.
    
    Example:
    --------
        with LoggerContext("training", epoch=1, batch=100):
            logger.info("Processing batch")
    """
    
    def __init__(self, context: str, **kwargs):
        self.context = context
        self.kwargs = kwargs
        self._token = None
    
    def __enter__(self):
        self._token = logger.contextualize(**{
            "context": self.context,
            **self.kwargs
        })
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token:
            self._token.__exit__(exc_type, exc_val, exc_tb)


# Convenience functions for common logging patterns
def log_model_params(model_name: str, params: dict):
    """Log model parameters in a structured way."""
    logger.info(f"Model: {model_name}")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")


def log_metrics(metrics: dict, prefix: str = ""):
    """Log metrics in a structured way."""
    prefix_str = f"{prefix}/" if prefix else ""
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{prefix_str}{key}: {value:.4f}")
        else:
            logger.info(f"{prefix_str}{key}: {value}")


def log_data_stats(data_name: str, shape: tuple, dtype: str = None):
    """Log data statistics."""
    msg = f"Data '{data_name}': shape={shape}"
    if dtype:
        msg += f", dtype={dtype}"
    logger.info(msg)


# Initialize default logger
setup_logger()
