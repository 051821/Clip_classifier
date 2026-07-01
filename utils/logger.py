"""
utils/logger.py
---------------
Centralised logger used across all modules.
"""

import logging
import sys


def get_logger(name: str = "clip_classifier") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler("classifier.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
