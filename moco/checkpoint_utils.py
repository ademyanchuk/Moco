import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from timm.utils import get_state_dict, unwrap_model


def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    models_path: Path,
    exp_name: str,
) -> None:
    save_state = {
        "epoch": epoch + 1,  # increment epoch (to not repeat then resume)
        "state_dict": get_state_dict(model, unwrap_model),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(
        save_state, f"{models_path}/{exp_name}.pth",
    )


def resume_checkpoint(
    model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True
):
    resume_epoch = None
    best_loss = None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        if log_info:
            logging.info("Restoring model state from checkpoint...")
        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            name = k[7:] if k.startswith("module") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        if optimizer is not None and "optimizer" in checkpoint:
            if log_info:
                logging.info("Restoring optimizer state from checkpoint...")
            optimizer.load_state_dict(checkpoint["optimizer"])

        if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
            if log_info:
                logging.info("Restoring AMP loss scaler state from checkpoint...")
            loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

        if "epoch" in checkpoint:
            resume_epoch = checkpoint["epoch"]
        if "val_loss" in checkpoint:
            best_loss = checkpoint["val_loss"]

        if log_info:
            logging.info(
                "Loaded checkpoint '{}' (epoch {})".format(
                    checkpoint_path, resume_epoch
                )
            )
    else:
        model.load_state_dict(checkpoint)
        if log_info:
            logging.info("Loaded checkpoint '{}'".format(checkpoint_path))
    return resume_epoch, best_loss
