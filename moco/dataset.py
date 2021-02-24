"""Data is from NihChestXRay dataset https://www.kaggle.com/nih-chest-xrays/data
   Dataset implementation is specific for this exact data.
"""

from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

AlbuAug = A.core.composition.Compose


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class NihChestPair(Dataset):
    def __init__(
        self, root_dir: Path, transform: AlbuAug,
    ):
        self.root_dir = root_dir
        self.transform = transform

        self._img_paths = self._get_img_paths()
        assert len(self._img_paths) > 0, f"No images found in dir: {self.root_dir}"

    def _get_img_paths(self):
        """NihChestXRay dataset specific."""
        sub_dirs = [f / "images" for f in self.root_dir.iterdir() if f.is_dir()]
        file_paths = []
        for dir in sub_dirs:
            file_paths += list(dir.glob("*.png"))
        return file_paths

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx: int):

        file_path = self._img_paths[idx]

        # here input will be greyscale
        image = Image.open(file_path).convert("RGB")  # make it 3 channels
        image = np.array(image)

        # ToTensor and Normalise required in all cases
        im_1 = self.transform(image=image)["image"]
        im_2 = self.transform(image=image)["image"]

        return im_1, im_2


def get_transform(crop_size: int):
    return A.Compose(
        [
            A.RandomResizedCrop(crop_size, crop_size, scale=(0.2, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=0, sigma_limit=(0.1, 2.0), p=0.5),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ],
    )
