"""Structured logging setup."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log") -> None:
    """Configure root logger with console + file output."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    formatter = logging.Formatter(fmt)
    
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper()))
    root.handlers.clear()
    
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)
    
    file_h = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setFormatter(formatter)
    root.addHandler(file_h)
    
    for noisy in ("httpx", "urllib3", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger inheriting root config."""
    return logging.getLogger(name)