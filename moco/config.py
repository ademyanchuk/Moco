"""Configuration file"""
from pathlib import Path

DATA_ROOT = Path("~/Data/NIHChest").expanduser()

Config = {
    "data_root": str(DATA_ROOT),
    "crop_size": 512,
}
