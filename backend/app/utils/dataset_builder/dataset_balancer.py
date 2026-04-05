"""
DatasetBalancer module - handles class balancing for FAKE/REAL labels.
"""

import random
import logging
from typing import List, Optional
from collections import defaultdict

from core.schemas import FaceCSVItem


class DatasetBalancer:
    """
    Handles balancing of FAKE and REAL samples in the dataset.

    Supports two balancing methods:
    - undersample: Reduce majority class to match minority class
    - oversample: Duplicate minority class to match majority class

    Maintains video integrity during undersampling to avoid splitting videos
    across balanced/unbalanced portions.
    """

    def __init__(self, random_state: Optional[int] = 42):
        """
        Initialize DatasetBalancer.

        Args:
            random_state: Random seed for reproducibility. None for no seeding.
        """
        self.random_state = random_state
        self.logger = logging.getLogger("DatasetBalancer")

        if random_state is not None:
            random.seed(random_state)

    def balance(
        self,
        faces_data: List[FaceCSVItem],
        method: str = 'undersample'
    ) -> List[FaceCSVItem]:
        """
        Balance the dataset to have equal number of FAKE and REAL samples.

        Args:
            faces_data: List of face items to balance
            method: Balancing method - 'undersample' or 'oversample'

        Returns:
            Balanced list of FaceCSVItem instances

        Raises:
            ValueError: If method is not 'undersample' or 'oversample'
        """
        if method not in ['undersample', 'oversample']:
            raise ValueError(
                f"Method must be 'undersample' or 'oversample', got '{method}'"
            )

        # Group by label
        fake_items = [item for item in faces_data if item.Label == "FAKE"]
        real_items = [item for item in faces_data if item.Label == "REAL"]

        fake_count = len(fake_items)
        real_count = len(real_items)

        self.logger.info(f"Before balancing: FAKE={fake_count}, REAL={real_count}")

        if fake_count == real_count:
            self.logger.info("Dataset already balanced")
            return faces_data

        # Determine majority and minority classes
        if fake_count > real_count:
            majority_items = fake_items
            minority_items = real_items
        else:
            majority_items = real_items
            minority_items = fake_items

        # Determine target count based on method
        if method == 'undersample':
            # For undersampling, reduce majority to minority size
            target_count = len(minority_items)
            balanced_data = self._undersample(
                majority_items, minority_items, target_count
            )
        else:  # oversample
            # For oversampling, increase minority to majority size
            target_count = len(majority_items)
            balanced_data = self._oversample(
                majority_items, minority_items, target_count
            )

        # Shuffle final dataset
        random.shuffle(balanced_data)

        fake_final = sum(1 for x in balanced_data if x.Label == 'FAKE')
        real_final = sum(1 for x in balanced_data if x.Label == 'REAL')

        self.logger.info(
            f"After balancing ({method}): FAKE={fake_final}, REAL={real_final}"
        )

        return balanced_data

    def _undersample(
        self,
        majority_items: List[FaceCSVItem],
        minority_items: List[FaceCSVItem],
        target_count: int
    ) -> List[FaceCSVItem]:
        """
        Undersample majority class while maintaining video integrity.

        Videos are selected randomly until target count is reached.
        This ensures no video is split between selected/unselected portions.

        Args:
            majority_items: Items from majority class
            minority_items: Items from minority class
            target_count: Target number of samples for majority class

        Returns:
            Balanced dataset with undersampled majority class
        """
        # Group majority items by video to maintain video integrity
        video_groups = defaultdict(list)
        for item in majority_items:
            video_groups[item.Video_Path].append(item)

        # Randomly select videos until we reach target count
        selected_items = []
        video_paths = list(video_groups.keys())
        random.shuffle(video_paths)

        for video_path in video_paths:
            video_items = video_groups[video_path]

            # If adding this video doesn't exceed target, add all its faces
            if len(selected_items) + len(video_items) <= target_count:
                selected_items.extend(video_items)
            # If we're close to target, add partial video to reach exact target
            elif len(selected_items) < target_count:
                remaining = target_count - len(selected_items)
                selected_items.extend(random.sample(video_items, remaining))
                break
            else:
                break

        return minority_items + selected_items

    def _oversample(
        self,
        majority_items: List[FaceCSVItem],
        minority_items: List[FaceCSVItem],
        target_count: int
    ) -> List[FaceCSVItem]:
        """
        Oversample minority class by duplicating samples.

        Args:
            majority_items: Items from majority class
            minority_items: Items from minority class
            target_count: Target number of samples for minority class

        Returns:
            Balanced dataset with oversampled minority class
        """
        # Calculate how many times to duplicate and remainder
        times_to_duplicate = target_count // len(minority_items)
        remainder = target_count % len(minority_items)

        # Duplicate minority items
        oversampled_minority = (
            minority_items * times_to_duplicate +
            random.sample(minority_items, remainder)
        )

        return majority_items + oversampled_minority

    def get_class_distribution(self, faces_data: List[FaceCSVItem]) -> dict:
        """
        Get the distribution of FAKE and REAL labels in the dataset.

        Args:
            faces_data: List of face items

        Returns:
            Dictionary with counts and percentages for each label
        """
        fake_count = sum(1 for item in faces_data if item.Label == "FAKE")
        real_count = sum(1 for item in faces_data if item.Label == "REAL")
        total = len(faces_data)

        return {
            "fake": {
                "count": fake_count,
                "percentage": fake_count / total * 100 if total > 0 else 0
            },
            "real": {
                "count": real_count,
                "percentage": real_count / total * 100 if total > 0 else 0
            },
            "total": total,
            "ratio": fake_count / real_count if real_count > 0 else float('inf')
        }
