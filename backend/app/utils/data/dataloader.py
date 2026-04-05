"""
DataLoader module for batching face images.

Uses PyTorch's DataLoader for efficient multi-worker data loading.
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader
import logging
from typing import Optional

from data.dataset import FaceDataset


def create_dataloader(
    dataset: FaceDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> TorchDataLoader:
    """
    Create a PyTorch DataLoader for the given dataset.

    Args:
        dataset: FaceDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch

    Returns:
        PyTorch DataLoader instance
    """
    logger = logging.getLogger("data.dataloader")

    # Disable pin_memory if no CUDA
    if not torch.cuda.is_available():
        pin_memory = False

    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )

    logger.info(
        f"Created DataLoader: batch_size={batch_size}, shuffle={shuffle}, "
        f"num_workers={num_workers}, total_batches={len(loader)}"
    )

    return loader


class DataLoaderWrapper:
    """
    Wrapper around PyTorch DataLoader with device placement.

    Automatically moves batches to the specified device.
    """

    def __init__(
        self,
        dataset: FaceDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        device: Optional[torch.device] = None,
        drop_last: bool = False
    ):
        """
        Initialize DataLoaderWrapper.

        Args:
            dataset: FaceDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            device: Target device for tensors
            drop_last: Drop last incomplete batch
        """
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.logger = logging.getLogger("data.DataLoaderWrapper")

        pin_memory = torch.cuda.is_available() and self.device.type == "cuda"

        self._loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0
        )

        self.logger.info(
            f"DataLoaderWrapper initialized: {len(dataset)} samples, "
            f"{len(self._loader)} batches, device={self.device}"
        )

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        for images, labels in self._loader:
            yield images.to(self.device), labels.to(self.device)


# Backward compatibility
DataLoader = DataLoaderWrapper
