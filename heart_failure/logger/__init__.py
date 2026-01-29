import logging
import os
from datetime import datetime
from from_root import from_root


LOG_DIR = "logs"

LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

LOG_FILE_PATH = os.path.join(
    from_root(),
    LOG_DIR,
    LOG_FILE_NAME
)


# Create log directory safely
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.DEBUG,
    format="[ %(asctime)s ] %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("heart_failure_logger")