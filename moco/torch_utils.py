import os
import random

import numpy as np
import torch


def seed_torch(seed=1982):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using pytorch device: {device}")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_properties(device))
    return device
