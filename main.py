"""Entry point to do contrastive pretraining with Moco framework.
   Trainig is only on one GPU (no multi-gpu support)
   Make sure you have a source code in `~/Projects/Moco`
   and data in `~/Data/NIHChest`
"""


import logging
from moco.train import train
from moco.config import DATA_ROOT, LOGS_PATH, MODELS_PATH, Config, save_config_yaml
from moco.log_utils import experiment_name, get_commit_hash, setup_logg
from moco.torch_utils import seed_torch, set_device


def main():
    # Setup
    seed_torch()
    device = set_device()
    exp_name = experiment_name(Config["debug"])
    # Check all folders exist
    for path in (DATA_ROOT, LOGS_PATH, MODELS_PATH):
        assert path.exists()
    # Save config for experiment reproducibility
    save_config_yaml(Config, exp_name)

    # If resume training experiment
    resume_path = None
    # Resume Experiment is provided
    if Config["resume"]:
        exp_name = Config["resume"]
        resume_path = MODELS_PATH / f"{exp_name}.pth"
        assert resume_path.exists()

    _ = setup_logg(LOGS_PATH, exp_name)
    commit_hash = get_commit_hash()
    logging.info(f"Experiment: {exp_name}, git commit: {commit_hash}")

    train(DATA_ROOT, MODELS_PATH, exp_name, Config, resume_path, device)


if __name__ == "__main__":
    main()
