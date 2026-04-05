"""
Dataset module for loading face images for deepfake detection.
"""

import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import logging

from core.schemas import PreprocessCSVItem
from utils.csv_services import CSVService


class FaceDataset(TorchDataset):
    """
    PyTorch Dataset for loading preprocessed face images.

    Loads images from CSV metadata and applies transforms for model input.
    Labels: FAKE -> 0, REAL -> 1
    """

    LABEL_MAP = {"FAKE": 0, "REAL": 1}

    def __init__(
        self,
        csv_path: str,
        data_root: Optional[str] = None,
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = False
    ):
        """
        Initialize FaceDataset.

        Args:
            csv_path: Path to CSV file with face metadata
            data_root: Root directory for face images. If None, uses csv_path parent.
            image_size: Target image size (height, width)
            augment: Whether to apply data augmentation (for training)
        """
        self.logger = logging.getLogger("data.FaceDataset")
        self.csv_path = Path(csv_path)

        # Data root is the directory containing the CSV
        if data_root is None:
            self.data_root = self.csv_path.parent
        else:
            self.data_root = Path(data_root)

        self.image_size = image_size
        self.augment = augment

        # Load CSV metadata
        self.items: list[PreprocessCSVItem] = CSVService.load_csv(
            str(self.csv_path), PreprocessCSVItem
        )

        # Build transforms
        self.transform = self._build_transforms()

        self.logger.info(
            f"Loaded {len(self.items)} samples from {self.csv_path.name} "
            f"(augment={augment})"
        )

    def _build_transforms(self) -> transforms.Compose:
        """Build image transforms pipeline."""
        transform_list = []

        # Resize to target size
        transform_list.append(transforms.Resize(self.image_size))

        # Data augmentation for training
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05
                ),
            ])

        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Args:
            index: Sample index

        Returns:
            Tuple of (image_tensor, label)
            - image_tensor: Shape [3, H, W], normalized
            - label: 0 for FAKE, 1 for REAL
        """
        item = self.items[index]

        # Build full image path
        image_path = self.data_root / item.Face_Path

        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)

        # Convert label to integer
        label = self.LABEL_MAP[item.Label]

        return image_tensor, label

    def get_label_distribution(self) -> dict:
        """Get distribution of labels in dataset."""
        fake_count = sum(1 for item in self.items if item.Label == "FAKE")
        real_count = sum(1 for item in self.items if item.Label == "REAL")
        return {
            "FAKE": fake_count,
            "REAL": real_count,
            "total": len(self.items),
            "ratio": fake_count / real_count if real_count > 0 else float('inf')
        }


# Backward compatibility alias
Dataset = FaceDataset
