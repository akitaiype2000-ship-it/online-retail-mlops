import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "running_logs.log"),
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)