"""All training functions"""
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import NihChestPair, get_transform
from .model import ModelMoCo
from .optimizer import init_optimizer, init_scheduler
from .checkpoint_utils import resume_checkpoint, save_checkpoint


def train(
    data_root: Path,
    models_path: Path,
    exp_name: str,
    config: dict,
    resume_path: Path,
    device: torch.device,
):
    """One full training run"""
    train_dataset = NihChestPair(
        data_root, transform=get_transform(crop_size=config["crop_size"])
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    model = ModelMoCo(
        dim=config["moco_dim"],
        K=config["moco_K"],
        m=config["moco_m"],
        T=config["moco_T"],
        arch=config["moco_arch"],
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = init_optimizer(model.parameters(), config["opt_conf"])

    # optionally resume from a checkpoint
    start_epoch = 0
    resume_epoch = None
    if resume_path:
        resume_epoch, _ = resume_checkpoint(model, resume_path, optimizer=optimizer)
        start_epoch = resume_epoch

    scheduler = init_scheduler(optimizer, config["sch_conf"])
    if scheduler is not None and resume_epoch is not None:
        scheduler.step(resume_epoch)

    for epoch in range(start_epoch, config["num_epochs"]):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, epoch, config, device
        )
        logging.info(f"Epoch {epoch} - avg train loss: {train_loss:.4f}")
        # lr scheduler step
        if scheduler is not None:
            scheduler.step(epoch)
        # save
        save_checkpoint(epoch, model, optimizer, models_path, exp_name)


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: Any,
    optimizer: Any,
    epoch: int,
    config: dict,
    device: torch.device,
):
    model.train()
    total_loss, total_num = 0.0, 0
    train_bar = tqdm(data_loader)
    for im_1, im_2 in train_bar:
        im_1 = im_1.to(device, non_blocking=True)
        im_2 = im_2.to(device, non_blocking=True)

        output, target = model(im_q=im_1, im_k=im_2)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description(
            "Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}".format(
                epoch,
                config["num_epochs"],
                optimizer.param_groups[0]["lr"],
                total_loss / total_num,
            )
        )

    return total_loss / total_num
