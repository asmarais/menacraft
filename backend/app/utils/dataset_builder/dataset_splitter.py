"""
DatasetSplitter module - handles train/test/eval splitting with video integrity.
"""

import random
import logging
from typing import List, Optional
from collections import defaultdict

from core.schemas import FaceCSVItem, DatasetSplit


class DatasetSplitter:
    """
    Handles splitting of dataset into train/test/eval sets.

    Ensures that:
    1. Videos are never split between different sets (train/test/eval)
    2. Videos from the same source dataset stay together
    3. Each split maintains stratification by label (FAKE/REAL)

    This prevents data leakage between train/test/eval sets.
    """

    def __init__(self, random_state: Optional[int] = 42):
        """
        Initialize DatasetSplitter.

        Args:
            random_state: Random seed for reproducibility. None for no seeding.
        """
        self.random_state = random_state
        self.logger = logging.getLogger("DatasetSplitter")

        if random_state is not None:
            random.seed(random_state)

    def split(
        self,
        faces_data: List[FaceCSVItem],
        train_ratio: float = 0.7,
        test_ratio: float = 0.15,
        eval_ratio: float = 0.15
    ) -> DatasetSplit:
        """
        Split dataset into train/test/eval sets while keeping videos intact.

        Videos from the same dataset are never mixed between splits.
        This ensures no data leakage between train/test/eval sets.

        Args:
            faces_data: List of face items to split
            train_ratio: Proportion for training set (default: 0.7)
            test_ratio: Proportion for test set (default: 0.15)
            eval_ratio: Proportion for eval set (default: 0.15)

        Returns:
            DatasetSplit object containing train/test/eval splits

        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        # Validate ratios
        total_ratio = train_ratio + test_ratio + eval_ratio
        if not abs(total_ratio - 1.0) < 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {total_ratio:.6f}"
            )

        # Group by dataset and video to ensure no mixing
        dataset_video_groups = self._group_by_dataset_and_video(faces_data)

        train_items = []
        test_items = []
        eval_items = []

        # Process each dataset separately to avoid mixing
        for dataset_name, video_groups in dataset_video_groups.items():
            self.logger.info(
                f"Splitting dataset '{dataset_name}' with {len(video_groups)} videos"
            )

            # Get all video paths and shuffle them
            video_paths = list(video_groups.keys())
            random.shuffle(video_paths)

            # Group videos by label for stratified splitting
            fake_videos = [
                vp for vp in video_paths
                if video_groups[vp][0].Label == "FAKE"
            ]
            real_videos = [
                vp for vp in video_paths
                if video_groups[vp][0].Label == "REAL"
            ]

            # Split each label group separately (stratified split)
            for video_list in [fake_videos, real_videos]:
                train_vids, test_vids, eval_vids = self._split_videos(
                    video_list, train_ratio, test_ratio
                )

                # Add all faces from selected videos
                for vp in train_vids:
                    train_items.extend(video_groups[vp])
                for vp in test_vids:
                    test_items.extend(video_groups[vp])
                for vp in eval_vids:
                    eval_items.extend(video_groups[vp])

        # Shuffle each split
        random.shuffle(train_items)
        random.shuffle(test_items)
        random.shuffle(eval_items)

        split = DatasetSplit(
            train=train_items,
            test=test_items,
            eval=eval_items
        )

        # Log statistics
        self._log_split_statistics(split)

        return split

    def _group_by_dataset_and_video(
        self,
        faces_data: List[FaceCSVItem]
    ) -> dict:
        """
        Group face items by dataset and video path.

        Args:
            faces_data: List of face items

        Returns:
            Nested dictionary: {dataset_name: {video_path: [face_items]}}
        """
        dataset_video_groups = defaultdict(lambda: defaultdict(list))

        for item in faces_data:
            dataset_video_groups[item.Dataset][item.Video_Path].append(item)

        return dataset_video_groups

    def _split_videos(
        self,
        video_list: List[str],
        train_ratio: float,
        test_ratio: float
    ) -> tuple:
        """
        Split a list of video paths into train/test/eval sets.

        Args:
            video_list: List of video paths to split
            train_ratio: Proportion for training set
            test_ratio: Proportion for test set

        Returns:
            Tuple of (train_videos, test_videos, eval_videos)
        """
        n_videos = len(video_list)
        n_train = int(n_videos * train_ratio)
        n_test = int(n_videos * test_ratio)

        train_videos = video_list[:n_train]
        test_videos = video_list[n_train:n_train + n_test]
        eval_videos = video_list[n_train + n_test:]

        return train_videos, test_videos, eval_videos

    def _log_split_statistics(self, split: DatasetSplit) -> None:
        """
        Log statistics about the dataset split.

        Args:
            split: DatasetSplit object to analyze
        """
        stats = split.get_statistics()

        self.logger.info("Dataset split completed:")
        self.logger.info(
            f"  Train: {stats['train']['total']} samples "
            f"(FAKE: {stats['train']['fake']}, REAL: {stats['train']['real']})"
        )
        self.logger.info(
            f"  Test: {stats['test']['total']} samples "
            f"(FAKE: {stats['test']['fake']}, REAL: {stats['test']['real']})"
        )
        self.logger.info(
            f"  Eval: {stats['eval']['total']} samples "
            f"(FAKE: {stats['eval']['fake']}, REAL: {stats['eval']['real']})"
        )

    def validate_split(self, split: DatasetSplit, original_data: List[FaceCSVItem]) -> bool:
        """
        Validate that the split maintains data integrity.

        Checks:
        1. No duplicate faces across splits
        2. All original faces are present in one split
        3. No videos are split between different sets

        Args:
            split: DatasetSplit object to validate
            original_data: Original list of face items

        Returns:
            True if split is valid, False otherwise
        """
        all_split_faces = split.train + split.test + split.eval
        all_split_paths = {item.Face_Path for item in all_split_faces}
        original_paths = {item.Face_Path for item in original_data}

        # Check that all original faces are present
        if all_split_paths != original_paths:
            self.logger.error("Split doesn't contain all original faces")
            return False

        # Check for duplicates across splits
        train_paths = {item.Face_Path for item in split.train}
        test_paths = {item.Face_Path for item in split.test}
        eval_paths = {item.Face_Path for item in split.eval}

        if train_paths & test_paths:
            self.logger.error("Duplicate faces found between train and test")
            return False
        if train_paths & eval_paths:
            self.logger.error("Duplicate faces found between train and eval")
            return False
        if test_paths & eval_paths:
            self.logger.error("Duplicate faces found between test and eval")
            return False

        # Check that videos are not split
        train_videos = {item.Video_Path for item in split.train}
        test_videos = {item.Video_Path for item in split.test}
        eval_videos = {item.Video_Path for item in split.eval}

        if train_videos & test_videos:
            self.logger.error("Videos are split between train and test")
            return False
        if train_videos & eval_videos:
            self.logger.error("Videos are split between train and eval")
            return False
        if test_videos & eval_videos:
            self.logger.error("Videos are split between test and eval")
            return False

        self.logger.info("Split validation passed")
        return True
