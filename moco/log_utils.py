import logging
import subprocess
from datetime import datetime
from pathlib import Path


def setup_logg(logs_path: Path, name: str) -> logging.RootLogger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_format = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.handlers = [stream_handler]
    # add logging handler to save logs to the file
    log_fname = f"{logs_path}/{name}.log"
    file_handler = logging.FileHandler(log_fname, mode="a")
    file_handler.setFormatter(log_format)
    logger.handlers.append(file_handler)
    return logger


def get_timestamp() -> str:
    now = datetime.now()
    return now.strftime("%d-%m-%Y-%H-%M")


def get_commit_hash() -> str:
    commit_hash = subprocess.check_output(["git", "describe", "--always"]).strip()
    return commit_hash.decode()


def experiment_name(debug: bool):
    now = datetime.now()
    if debug:
        exp_name = "debug_" + now.strftime("%d-%m-%Y-%H-%M")
    else:
        exp_name = "exp_" + now.strftime("%d-%m-%Y-%H-%M")
    return exp_name
