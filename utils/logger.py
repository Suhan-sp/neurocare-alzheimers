"""
Logger Utility Module
Provides standardized logging for training pipelines, feature extraction, and inference.
"""

import logging
import sys
import os

def setup_logger(name="alzheimer_system", log_file="system.log", level=logging.INFO):
    """
    Sets up a logger with both console and file handlers.
    
    Args:
        name (str): Name of the logger.
        log_file (str): File path to store logs.
        level (int): Logging level.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger is already initialized
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not set up file logger at {log_file}: {e}")
        
    return logger

# Default global logger
logger = setup_logger()
