"""Configuration file"""
from pathlib import Path

import yaml

DATA_ROOT = Path("~/Data/NIHChest").expanduser()
PROJ_PATH = Path("~/Projects/Moco").expanduser()
LOGS_PATH = PROJ_PATH / "logs"
MODELS_PATH = PROJ_PATH / "models"

Config = {
    "debug": True,
    "resume": "",  # empty string or exisiting experiment name
    "crop_size": 256,
    "batch_size": 16,
    "moco_dim": 128,
    "moco_K": 4096,
    "moco_m": 0.999,
    "moco_T": 0.1,
    "moco_arch": "resnet50d",
    "opt_conf": {"adamw": {"lr": 5e-4, "weight_decay": 0.0}},
    "sch_conf": {
        "cosine": {
            "t_initial": 2,
            "lr_min": 5e-8,
            "warmup_t": 0,
            "warmup_lr_init": 5e-7,
        }
    },
    "num_epochs": 2,
}


def create_folders():
    """Helper to create all necessary folders"""
    for p in [DATA_ROOT, LOGS_PATH, MODELS_PATH]:
        p.mkdir(parents=True, exist_ok=True)


def save_config_yaml(config: dict, exp_name: str):
    with open(LOGS_PATH / f"{exp_name}_conf.yaml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)
