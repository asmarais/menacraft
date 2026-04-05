"""
FaceDatasetBuilder module for extracting faces from frames.
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Tuple, Any
from multiprocessing import Pool
from functools import partial

import cv2
from tqdm import tqdm

from preprocessing.face_detector import FaceDetector
from preprocessing.face_extractor import FaceExtractor
from core.schemas import FrameCSVItem, FaceCSVItem
from utils.csv_services import CSVService


class FaceDatasetBuilder:
    """
    Handles face extraction from frames in parallel.

    This class is responsible for:
    - Detecting faces in frame images
    - Extracting and preprocessing face regions
    - Saving face images to disk
    - Generating and saving face metadata CSV
    """

    FACES_METADATA_FILE = "faces_metadata.csv"
    FRAMES_METADATA_FILE = "frames_metadata.csv"

    def __init__(
        self,
        frame_dir: str,
        faces_dir: str,
        num_workers: int,
        min_face_size: int = 40,
        confidence_threshold: float = 0.7,
        only_keep_top_face: bool = True,
        face_target_size: Tuple[int, int] = (224, 224),
        face_normalization: str = "zero_one",
        face_padding: int = 20,
        device: str = "cpu"
    ):
        """
        Initialize FaceDatasetBuilder.

        Args:
            frame_dir: Directory containing extracted frames
            faces_dir: Directory to save extracted faces
            num_workers: Number of parallel workers
            min_face_size: Minimum face size for detection
            confidence_threshold: Minimum confidence for face detection
            only_keep_top_face: Whether to keep only the most confident face
            face_target_size: Target size for face images
            face_normalization: Normalization method
            face_padding: Padding in pixels around face bbox
            device: Device to use for face detection (cpu or cuda)
        """
        self.frame_dir = frame_dir
        self.faces_dir = faces_dir
        self.num_workers = num_workers

        # Face detection/extraction config
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self.only_keep_top_face = only_keep_top_face
        self.face_target_size = face_target_size
        self.face_normalization = face_normalization
        self.face_padding = face_padding
        self.device = device

        self.logger = logging.getLogger("FaceDatasetBuilder")

    def extract_faces(self) -> List[FaceCSVItem]:
        """
        Extract faces from frames in parallel.

        Returns:
            List of FaceCSVItem instances
        """
        self.logger.info("Starting face extraction...")

        # Load frame metadata
        frames_data = self._load_frames_metadata()
        self.logger.info(f"Found {len(frames_data)} frames to process")

        # Process in parallel
        faces_csv_data, stats = self._parallel_process_frames(frames_data)

        # Save metadata
        self._save_metadata(faces_csv_data)

        # Log stats
        self._log_stats(stats, len(frames_data))

        return faces_csv_data

    def _load_frames_metadata(self) -> List[FrameCSVItem]:
        """Load frames metadata from CSV file."""
        metadata_path = os.path.join(self.frame_dir, self.FRAMES_METADATA_FILE)

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Frames metadata not found at {metadata_path}. "
                "Please run extract_frame() first."
            )

        return CSVService.load_csv(metadata_path, FrameCSVItem)

    def _parallel_process_frames(
        self,
        frames_data: List[FrameCSVItem]
    ) -> Tuple[List[FaceCSVItem], Dict[str, int]]:
        """
        Process frames to extract faces.

        Uses sequential processing for CUDA (GPU already parallelizes),
        multiprocessing for CPU.
        """
        if self.device == "cuda":
            # GPU: sequential processing (GPU is already massively parallel)
            self.logger.info("Using sequential processing with GPU acceleration")
            results = self._sequential_process_frames(frames_data)
        else:
            # CPU: multiprocessing for parallelization
            self.logger.info(f"Using multiprocessing with {self.num_workers} workers")
            results = self._multiprocess_frames(frames_data)

        # Flatten results and calculate stats
        faces_csv_data = []
        total_faces = 0
        frames_with_no_faces = 0

        for face_list in results:
            if face_list:
                faces_csv_data.extend(face_list)
                total_faces += len(face_list)
            else:
                frames_with_no_faces += 1

        stats = {
            'total_faces': total_faces,
            'frames_with_no_faces': frames_with_no_faces
        }

        return faces_csv_data, stats

    def _sequential_process_frames(
        self,
        frames_data: List[FrameCSVItem]
    ) -> List[List[FaceCSVItem]]:
        """
        Process frames sequentially (for GPU).

        Initializes detectors once and reuses them for all frames.
        """
        # Initialize detectors ONCE (this is the key optimization for GPU)
        self.logger.info("Initializing face detector and extractor...")
        face_detector = FaceDetector(
            min_face_size=self.min_face_size,
            confidence_threshold=self.confidence_threshold,
            only_keep_top=self.only_keep_top_face,
            device=self.device
        )
        face_extractor = FaceExtractor(
            target_size=self.face_target_size,
            normalization=self.face_normalization,
            padding=self.face_padding
        )

        # Process all frames with the same detector instances
        results = []
        for frame_data in tqdm(
            frames_data,
            desc="Extracting faces (GPU)",
            unit="frame"
        ):
            result = self._process_single_frame_with_detectors(
                frame_data,
                face_detector,
                face_extractor
            )
            results.append(result)

        return results

    def _multiprocess_frames(
        self,
        frames_data: List[FrameCSVItem]
    ) -> List[List[FaceCSVItem]]:
        """Process frames in parallel using multiprocessing (for CPU)."""
        process_func = partial(
            self._process_single_frame,
            faces_dir=str(self.faces_dir),
            min_face_size=self.min_face_size,
            confidence_threshold=self.confidence_threshold,
            only_keep_top_face=self.only_keep_top_face,
            face_target_size=self.face_target_size,
            face_normalization=self.face_normalization,
            face_padding=self.face_padding,
            device=self.device
        )

        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, frames_data),
                total=len(frames_data),
                desc="Extracting faces (CPU)",
                unit="frame"
            ))

        return results

    def _process_single_frame_with_detectors(
        self,
        frame_data: FrameCSVItem,
        face_detector: FaceDetector,
        face_extractor: FaceExtractor
    ) -> List[FaceCSVItem]:
        """
        Process a single frame using pre-initialized detectors.

        This method is used for GPU processing where we want to reuse
        the same detector instances across all frames.

        Args:
            frame_data: FrameCSVItem containing frame metadata
            face_detector: Pre-initialized FaceDetector instance
            face_extractor: Pre-initialized FaceExtractor instance

        Returns:
            List of FaceCSVItem instances
        """
        # Reconstruct absolute path from relative path
        frame_dir = os.path.join(os.path.dirname(self.faces_dir), "frames")
        frame_path = os.path.join(frame_dir, frame_data.Frame_Path)

        frame_number = frame_data.Frame_Number
        video_path = frame_data.Video_Path
        label = frame_data.Label
        dataset = frame_data.Dataset

        if not os.path.exists(frame_path):
            return []

        try:
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                return []

            # Detect faces (using pre-initialized detector)
            detections = face_detector.detect_faces(frame, return_landmarks=False)

            if not detections:
                return []

            # Extract and save each face
            faces_data = []
            for idx, detection in enumerate(detections):
                face_item = self._extract_and_save_face(
                    frame=frame,
                    detection=detection,
                    frame_number=frame_number,
                    frame_path=frame_path,
                    video_path=video_path,
                    label=label,
                    dataset=dataset,
                    idx=idx,
                    faces_dir=str(self.faces_dir),
                    face_extractor=face_extractor
                )
                faces_data.append(face_item)

            return faces_data

        except Exception as e:
            self.logger.error(f"Error processing frame {frame_path}: {str(e)}")
            return []

    @staticmethod
    def _process_single_frame(
        frame_data: FrameCSVItem,
        faces_dir: str,
        min_face_size: int,
        confidence_threshold: float,
        only_keep_top_face: bool,
        face_target_size: Tuple[int, int],
        face_normalization: str,
        face_padding: int,
        device: str = "cpu"
    ) -> List[FaceCSVItem]:
        """
        Worker function to extract faces from a single frame.

        Args:
            frame_data: FrameCSVItem containing frame metadata
            faces_dir: Directory to save extracted faces
            device: Device to use for face detection (cpu or cuda)
            (other args are configuration parameters)

        Returns:
            List of FaceCSVItem instances
        """
        # Reconstruct absolute path from relative path
        # frame_data.Frame_Path is relative to frame_dir
        # We need to get frame_dir from faces_dir (both are siblings)
        frame_dir = os.path.join(os.path.dirname(faces_dir), "frames")
        frame_path = os.path.join(frame_dir, frame_data.Frame_Path)

        frame_number = frame_data.Frame_Number
        video_path = frame_data.Video_Path
        label = frame_data.Label
        dataset = frame_data.Dataset

        if not os.path.exists(frame_path):
            return []

        try:
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                return []

            # Initialize detectors
            face_detector = FaceDetector(
                min_face_size=min_face_size,
                confidence_threshold=confidence_threshold,
                only_keep_top=only_keep_top_face,
                device=device
            )
            face_extractor = FaceExtractor(
                target_size=face_target_size,
                normalization=face_normalization,
                padding=face_padding
            )

            # Detect faces
            detections = face_detector.detect_faces(frame, return_landmarks=False)

            if not detections:
                return []

            # Extract and save each face
            faces_data = []
            for idx, detection in enumerate(detections):
                face_item = FaceDatasetBuilder._extract_and_save_face(
                    frame=frame,
                    detection=detection,
                    frame_number=frame_number,
                    frame_path=frame_path,
                    video_path=video_path,
                    label=label,
                    dataset=dataset,
                    idx=idx,
                    faces_dir=faces_dir,
                    face_extractor=face_extractor
                )
                faces_data.append(face_item)

            return faces_data

        except Exception as e:
            logger = logging.getLogger("FaceDatasetBuilder.Worker")
            logger.error(f"Error processing frame {frame_path}: {str(e)}")
            return []

    @staticmethod
    def _extract_and_save_face(
        frame: Any,
        detection: Dict[str, Any],
        frame_number: int,
        frame_path: str,
        video_path: str,
        label: str,
        dataset: str,
        idx: int,
        faces_dir: str,
        face_extractor: FaceExtractor
    ) -> FaceCSVItem:
        """Extract, save, and create metadata for a single face."""
        bbox = detection['bbox']
        confidence = detection['confidence']

        # Extract and preprocess face
        preprocessed_face, _ = face_extractor.extract_and_preprocess(
            frame, bbox, normalize=True
        )

        # Determine output path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        face_output_dir = os.path.join(faces_dir, video_name)
        os.makedirs(face_output_dir, exist_ok=True)

        face_filename = f"face_frame_{frame_number:06d}_{idx:02d}.jpg"
        face_path = os.path.join(face_output_dir, face_filename)

        # Save face
        face_extractor.save(preprocessed_face, face_path, denormalize=True)

        # Create relative path (relative to faces_dir)
        relative_face_path = os.path.join(video_name, face_filename)

        # Create metadata as FaceCSVItem
        return FaceCSVItem(
            Face_Path=relative_face_path,
            Video_Path=video_path,
            Label=label,
            Frame_Number=frame_number,
            Dataset=dataset,
            Confidence=confidence,
            BBox_X=bbox[0],
            BBox_Y=bbox[1],
            BBox_Width=bbox[2],
            BBox_Height=bbox[3],
            Face_Width=face_extractor.target_size[0],
            Face_Height=face_extractor.target_size[1]
        )

    def _save_metadata(self, csv_data: List[FaceCSVItem]) -> None:
        """Save face extraction metadata to CSV file."""
        metadata_path = os.path.join(self.faces_dir, self.FACES_METADATA_FILE)

        self.logger.info(f"Saving faces metadata to {metadata_path}...")

        CSVService.save_csv(metadata_path, csv_data, FaceCSVItem)

        self.logger.info(f"✓ Metadata saved to {metadata_path}")

    def _log_stats(
        self,
        stats: Dict[str, int],
        total_frames: int
    ) -> None:
        """Log face extraction statistics."""
        self.logger.info("✓ Face extraction complete!")
        self.logger.info(
            f"✓ {stats['total_faces']} faces extracted from {total_frames} frames"
        )
        self.logger.info(
            f"✓ {stats['frames_with_no_faces']} frames had no detectable faces"
        )
