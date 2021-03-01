"""Convert moco checkpoint encoders into regular timm checkpoints
   for Ranzcr Kaggle Comp
"""
import argparse
from moco.checkpoint_utils import resume_checkpoint
from pathlib import Path
import torch
from timm.utils import get_state_dict, unwrap_model

from moco.config import Config
from moco.model import ModelMoCo

parser = argparse.ArgumentParser(
    "Convert moco pretrained model encoder to timm checkpoints"
)
parser.add_argument(
    "--moco-path", required=True, help="Path to the pretrained moco checkpoint"
)


def main():
    """Load the pretrained moco model from args.moco_path and save
       both moco encoders to the same folder"""
    args = parser.parse_args()
    checkpoint = Path(args.moco_path)
    assert checkpoint.exists()
    model = ModelMoCo(
        dim=Config["moco_dim"],
        K=Config["moco_K"],
        m=Config["moco_m"],
        T=Config["moco_T"],
        arch=Config["moco_arch"],
    )
    _ = resume_checkpoint(model, checkpoint)
    encoder_q = model.encoder_q.net
    encoder_q.reset_classifier(11)  # hard code
    save_state = {
        "state_dict": get_state_dict(encoder_q, unwrap_model),
    }
    torch.save(
        save_state, f"{checkpoint.parents[0]}/{checkpoint.stem}_q.pth",
    )


if __name__ == "__main__":
    main()
