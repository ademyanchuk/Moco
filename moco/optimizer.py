from typing import Any, Dict
import torch
from timm import scheduler, optim

# Optimizer configs
adamw_conf = {"adamw": {"lr": 5e-4, "weight_decay": 0.0}}
sgd_conf = {"sgd": {"lr": 3.75e-3, "weight_decay": 1e-4, "momentum": 0.9}}

# Scheduler configs
cosine_conf = {
    "cosine": {"t_initial": 60, "lr_min": 0.0, "warmup_t": 0, "warmup_lr_init": 5e-4}
}


def init_optimizer(model_params: Any, conf_dict: Dict):
    """
    Initialize optimizer according a `conf_dict` params
    Only one key on outter level of config dict is allowed
    """
    if "adamw" in conf_dict:
        kwargs = conf_dict["adamw"]
        return torch.optim.AdamW(model_params, **kwargs)
    elif "sgd" in conf_dict:
        kwargs = conf_dict["sgd"]
        return torch.optim.SGD(model_params, **kwargs)
    elif "radam" in conf_dict:
        kwargs = conf_dict["radam"]
        return optim.RAdam(model_params, **kwargs)
    elif "rmsproptf" in conf_dict:
        kwargs = conf_dict["rmsproptf"]
        return optim.RMSpropTF(model_params, **kwargs)
    else:
        raise NotImplementedError


def init_scheduler(optimizer: Any, conf_dict: Dict):
    """
    Initialize scheduler according a `conf_dict` params
    Only one key on outter level of config dict is allowed
    """
    if "step" in conf_dict:
        kwargs = conf_dict["step"]
        return scheduler.StepLRScheduler(optimizer, **kwargs)
    elif "cosine" in conf_dict:
        kwargs = conf_dict["cosine"]
        return scheduler.CosineLRScheduler(optimizer, **kwargs)
    else:
        raise NotImplementedError
