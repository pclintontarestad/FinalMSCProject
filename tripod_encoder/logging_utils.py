import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name="tripod", level=logging.INFO, log_dir="logs"):
    Path(log_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = Path(log_dir) / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers on reload
    if logger.handlers:
        return logger

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)

    # File handler
    fh = logging.FileHandler(logfile)
    fh.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info("Logger initialised")
    logger.info(f"Logging to: {logfile}")

    return logger
