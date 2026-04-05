

import torch
from pathlib import Path
from typing import Optional, Union
import logging

from core.model import DeepFakeDetector


def load_model(
    model_path: Union[str, Path],
    device: Union[str, torch.device],
    eval_mode: bool = True
) -> DeepFakeDetector:

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")


    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = DeepFakeDetector()

    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    if eval_mode:
        model.eval()


    return model