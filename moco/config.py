"""Configuration file"""
from pathlib import Path

import yaml

from .optimizer import sgd_conf, cosine_conf

DATA_ROOT = Path("~/Data/NIHChest").expanduser()
PROJ_PATH = Path("~/Projects/Moco").expanduser()
LOGS_PATH = PROJ_PATH / "logs"
MODELS_PATH = PROJ_PATH / "models"

Config = {
    "debug": False,
    "resume": "",  # empty string or exisiting experiment name
    "crop_size": 320,
    "batch_size": 32,
    "moco_dim": 128,
    "moco_K": 4096,
    "moco_m": 0.999,
    "moco_T": 0.2,
    "moco_arch": "resnet200d",
    "opt_conf": sgd_conf,
    "sch_conf": cosine_conf,
    "num_epochs": 290,
}


def create_folders():
    """Helper to create all necessary folders"""
    for p in [DATA_ROOT, LOGS_PATH, MODELS_PATH]:
        p.mkdir(parents=True, exist_ok=True)


def save_config_yaml(config: dict, exp_name: str):
    with open(LOGS_PATH / f"{exp_name}_conf.yaml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)
