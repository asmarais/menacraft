"""
FrameDatasetBuilder module for extracting frames from videos.

Supports both CPU and GPU-accelerated extraction via decord.
"""

from __future__ import annotations

import os
import logging
from typing import List, Tuple
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm

from core.schemas import CSVItem, FrameCSVItem
from utils.csv_services import CSVService

# Import optimized extractor with fallback
try:
    from preprocessing.optimized_frames_extractor import (
        OptimizedFramesExtractor,
        DECORD_AVAILABLE
    )
except ImportError:
    DECORD_AVAILABLE = False


class FrameDatasetBuilder:
    """
    Handles frame extraction from videos in parallel.

    This class is responsible for:
    - Extracting frames from video files (CPU or GPU-accelerated)
    - Saving frame images to disk with parallel I/O
    - Generating and saving frame metadata CSV

    Performance modes:
    - CPU: Uses multiprocessing with decord CPU backend
    - GPU: Uses single-process decord GPU (NVDEC) + multi-threaded saving
    """

    FRAMES_METADATA_FILE = "frames_metadata.csv"

    def __init__(
        self,
        data_folder: str,
        frame_dir: str,
        videos_path: dict,
        num_workers: int,
        fps: int = 1,
        device: str = "cpu",
        batch_size: int = 32,
        num_save_threads: int = 4
    ):
        """
        Initialize FrameDatasetBuilder.

        Args:
            data_folder: Base folder containing video files
            frame_dir: Directory to save extracted frames
            videos_path: Dictionary mapping dataset name to list of CSVItem
            num_workers: Number of parallel workers for CPU mode
            fps: Frames per second to extract
            device: Device for decoding - "cpu" or "cuda"/"gpu"
            batch_size: Frames to decode at once (for GPU mode)
            num_save_threads: Threads for parallel frame saving
        """
        self.data_folder = data_folder
        self.frame_dir = frame_dir
        self.videos_path = videos_path
        self.num_workers = num_workers
        self.fps = fps
        self.device = device.lower()
        self.batch_size = batch_size
        self.num_save_threads = num_save_threads
        self.logger = logging.getLogger("FrameDatasetBuilder")

        # Log extraction mode
        if DECORD_AVAILABLE:
            self.logger.info(f"Using optimized decord extractor (device: {self.device})")
        else:
            self.logger.warning("decord not available, using slower OpenCV extraction")

    def extract_frames(self) -> List[FrameCSVItem]:
        """
        Extract frames from all videos.

        Uses GPU mode if device is cuda/gpu (single-process, batched).
        Uses CPU mode with multiprocessing otherwise.

        Returns:
            List of FrameCSVItem instances
        """
        self.logger.info("Starting frame extraction...")

        # Collect all videos
        all_videos = self._collect_videos()
        self.logger.info(f"Found {len(all_videos)} videos to process")

        # Choose extraction strategy based on device
        if self.device in ("cuda", "gpu"):
            self.logger.info("Using GPU-accelerated extraction (single-process, batched)")
            csv_data = self._gpu_process_videos(all_videos)
        else:
            self.logger.info(f"Using CPU extraction with {self.num_workers} workers")
            csv_data = self._parallel_process_videos(all_videos)

        # Save metadata
        self._save_metadata(csv_data)

        self.logger.info(
            f"✓ Frame extraction complete! "
            f"{len(csv_data)} frames extracted from {len(all_videos)} videos"
        )

        return csv_data

    def _collect_videos(self) -> List[Tuple[str, CSVItem]]:
        """Collect all videos from loaded CSV files."""
        videos = []
        for csv_name, video_list in self.videos_path.items():
            for video in video_list:
                videos.append((csv_name, video))
        return videos

    def _gpu_process_videos(
        self,
        videos: List[Tuple[str, CSVItem]]
    ) -> List[FrameCSVItem]:
        """
        Process videos using GPU-accelerated decoding.

        GPU decoding doesn't parallelize well across processes (shared GPU),
        so we process videos sequentially but with batched frame decoding
        and multi-threaded I/O.
        """
        all_frames = []

        for video_data in tqdm(videos, desc="Extracting frames (GPU)", unit="video"):
            frames = self._process_single_video_optimized(
                video_data=video_data,
                data_folder=str(self.data_folder),
                frame_dir=str(self.frame_dir),
                fps=self.fps,
                device="gpu",
                batch_size=self.batch_size,
                num_save_threads=self.num_save_threads
            )
            all_frames.extend(frames)

        return all_frames

    def _parallel_process_videos(
        self,
        videos: List[Tuple[str, CSVItem]]
    ) -> List[FrameCSVItem]:
        """Process videos in parallel using CPU."""
        process_func = partial(
            self._process_single_video_optimized,
            data_folder=str(self.data_folder),
            frame_dir=str(self.frame_dir),
            fps=self.fps,
            device="cpu",
            batch_size=self.batch_size,
            num_save_threads=max(1, self.num_save_threads // self.num_workers)
        )

        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, videos),
                total=len(videos),
                desc="Extracting frames (CPU)",
                unit="video"
            ))

        # Flatten results
        return [frame for video_frames in results for frame in video_frames]

    @staticmethod
    def _process_single_video_optimized(
        video_data: Tuple[str, CSVItem],
        data_folder: str,
        frame_dir: str,
        fps: int,
        device: str = "cpu",
        batch_size: int = 32,
        num_save_threads: int = 4
    ) -> List[FrameCSVItem]:
        """
        Worker function to extract frames using optimized extractor.

        Uses decord for fast decoding + multi-threaded saving.

        Args:
            video_data: Tuple of (csv_name, video CSVItem object)
            data_folder: Base data folder path
            frame_dir: Directory to save extracted frames
            fps: Frames per second to extract
            device: Decoding device ("cpu" or "gpu")
            batch_size: Frames to decode at once
            num_save_threads: Threads for parallel saving

        Returns:
            List of FrameCSVItem instances
        """
        csv_name, video = video_data
        video_path = video.File_Path
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        # Prefix with dataset name to avoid collisions when same video name exists in different folders
        dataset_prefix = csv_name.replace('.csv', '')
        video_name = f"{dataset_prefix}_{video_basename}"
        complete_video_path = os.path.join(data_folder, video_path)

        if not os.path.exists(complete_video_path):
            return []

        try:
            # Use optimized extractor if available
            if DECORD_AVAILABLE:
                extractor = OptimizedFramesExtractor(
                    video_path=complete_video_path,
                    nb_fps=fps,
                    device=device,
                    batch_size=batch_size,
                    num_save_threads=num_save_threads
                )

                output_dir = os.path.join(frame_dir, video_name)

                # Use the optimized batch extract + save method
                saved_frames = extractor.extract_and_save_frames(
                    output_dir=output_dir,
                    filename_template="frame_{frame_number:06d}.jpg"
                )

                frames_data = []
                for saved_path, metadata in saved_frames:
                    frame_filename = os.path.basename(saved_path)
                    relative_frame_path = os.path.join(video_name, frame_filename)

                    frame_item = FrameCSVItem(
                        Frame_Path=relative_frame_path,
                        Video_Path=video_path,
                        Label=video.Label,
                        Width=extractor.width,
                        Height=extractor.height,
                        Frame_Number=metadata['frame_number'],
                        Timestamp=metadata['timestamp'],
                        Dataset=csv_name.replace('.csv', '')
                    )
                    frames_data.append(frame_item)

                return frames_data
            else:
                # Fallback to original extractor
                return FrameDatasetBuilder._process_single_video_legacy(
                    video_data, data_folder, frame_dir, fps
                )

        except Exception as e:
            logger = logging.getLogger("FrameDatasetBuilder.Worker")
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return []

    @staticmethod
    def _process_single_video_legacy(
        video_data: Tuple[str, CSVItem],
        data_folder: str,
        frame_dir: str,
        fps: int
    ) -> List[FrameCSVItem]:
        """
        Legacy worker function using OpenCV-based extractor.

        Used as fallback when decord is not available.
        """
        from preprocessing.frames_extractor import FramesExtractor

        csv_name, video = video_data
        video_path = video.File_Path
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        # Prefix with dataset name to avoid collisions when same video name exists in different folders
        dataset_prefix = csv_name.replace('.csv', '')
        video_name = f"{dataset_prefix}_{video_basename}"
        complete_video_path = os.path.join(data_folder, video_path)

        if not os.path.exists(complete_video_path):
            return []

        try:
            extractor = FramesExtractor(complete_video_path, nb_fps=fps)
            output_dir = os.path.join(frame_dir, video_name)
            os.makedirs(output_dir, exist_ok=True)

            frames_data = []
            for frame, metadata in extractor.extract_frames():
                frame_filename = f"frame_{metadata['frame_number']:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)

                extractor.save_frame(frame, frame_path)

                relative_frame_path = os.path.join(video_name, frame_filename)

                frame_item = FrameCSVItem(
                    Frame_Path=relative_frame_path,
                    Video_Path=video_path,
                    Label=video.Label,
                    Width=extractor.width,
                    Height=extractor.height,
                    Frame_Number=metadata['frame_number'],
                    Timestamp=metadata['timestamp'],
                    Dataset=csv_name.replace('.csv', '')
                )
                frames_data.append(frame_item)

            return frames_data

        except Exception as e:
            logger = logging.getLogger("FrameDatasetBuilder.Worker")
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return []

    def _save_metadata(self, csv_data: List[FrameCSVItem]) -> None:
        """Save frame extraction metadata to CSV file."""
        metadata_path = os.path.join(self.frame_dir, self.FRAMES_METADATA_FILE)

        self.logger.info(f"Saving metadata to {metadata_path}...")

        CSVService.save_csv(metadata_path, csv_data, FrameCSVItem)

        self.logger.info("✓ Metadata saved")
