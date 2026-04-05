"""
API Services

Business logic layer wrapping the inference engine for API use.
"""

import time
import uuid
import shutil
from pathlib import Path
from logging import getLogger, Logger
from typing import Optional

from inference.inference_engine import InferenceEngine
from utils.config_dataclasses import InferenceConfig
from utils.config_services import ConfigService
from utils.inference_results_dataclasses import VideoInferenceResult
from api.config import APISettings
from api.schemas import (
    AnalysisOptions,
    AnalysisResponse,
    AggregationMetadata,
    FrameResult,
)


class InferenceService:
    """Service layer for deepfake inference operations."""

    def __init__(self, settings: APISettings):
        """Initialize the inference service.

        Parameters:
            settings: API settings containing configuration paths
        """
        self.logger: Logger = getLogger("api.services")
        self.settings = settings
        self._engine: Optional[InferenceEngine] = None
        self._config: Optional[InferenceConfig] = None

    def _load_config(self) -> InferenceConfig:
        """Load inference configuration from file or use defaults.

        Returns:
            InferenceConfig: Loaded or default configuration
        """
        if self.settings.inference_config_path and self.settings.inference_config_path.exists():
            self.logger.info(f"Loading config from {self.settings.inference_config_path}")
            config_dict = ConfigService.load_config(self.settings.inference_config_path)
            return InferenceConfig.from_dict(config_dict)

        self.logger.info("Using default inference configuration")
        return InferenceConfig()

    def initialize(self) -> None:
        """Initialize the inference engine.

        This should be called during application startup.
        """
        self.logger.info("Initializing inference service...")
        self._config = self._load_config()
        self._engine = InferenceEngine(config=self._config)
        self.logger.info("Inference service initialized successfully")

    @property
    def engine(self) -> InferenceEngine:
        """Get the inference engine, initializing if needed."""
        if self._engine is None:
            self.initialize()
        return self._engine

    @property
    def is_ready(self) -> bool:
        """Check if the service is ready for inference."""
        return self._engine is not None

    @property
    def device(self) -> str:
        """Get the current inference device."""
        if self._engine is None:
            return "not_initialized"
        return str(self._engine.device)

    def analyze_video(
        self,
        video_path: Path,
        options: AnalysisOptions,
        include_frame_results: bool = False
    ) -> tuple[AnalysisResponse, Optional[Path]]:
        """Analyze a video for deepfake detection.

        Parameters:
            video_path: Path to the video file
            options: Analysis options
            include_frame_results: Whether to include per-frame results

        Returns:
            Tuple of (AnalysisResponse, Optional[Path to gradcam video])
        """
        start_time = time.time()

        gradcam_output_path: Optional[Path] = None
        if options.enable_gradcam:
            video_id = uuid.uuid4().hex[:8]
            gradcam_output_path = self.settings.output_dir / f"gradcam_{video_id}.mp4"

        self.logger.info(f"Starting video analysis: {video_path.name}")
        self.logger.info(f"Options: fps={options.frames_per_second}, "
                        f"aggregation={options.aggregation_method}, "
                        f"gradcam={options.enable_gradcam}")

        result: VideoInferenceResult = self.engine.predict_video(
            video_path=video_path,
            frames_per_second=options.frames_per_second,
            aggregation_method=options.aggregation_method,
            batch_size=options.batch_size,
            enable_gradcam=options.enable_gradcam,
            gradcam_output_path=gradcam_output_path,
            gradcam_alpha=options.gradcam_alpha
        )

        processing_time = time.time() - start_time

        authenticity_score = self._calculate_authenticity_score(result)

        aggregation_metadata = AggregationMetadata(
            method=result.aggregation_metadata.get("method", options.aggregation_method),
            frame_count=result.num_frames_analyzed,
            fake_ratio=result.aggregation_metadata.get("fake_ratio"),
            real_ratio=result.aggregation_metadata.get("real_ratio"),
            average_fake_confidence=result.aggregation_metadata.get("average_fake_confidence"),
            average_real_confidence=result.aggregation_metadata.get("average_real_confidence"),
        )

        frame_results_list: Optional[list[FrameResult]] = None
        if include_frame_results:
            frame_results_list = [
                FrameResult(
                    prediction=fr.prediction,
                    confidence=fr.confidence,
                    probabilities=fr.probabilities
                )
                for fr in result.frame_results
            ]

        response = AnalysisResponse(
            success=True,
            prediction=result.aggregate_prediction,
            confidence=result.aggregate_confidence,
            authenticity_score=authenticity_score,
            num_frames_analyzed=result.num_frames_analyzed,
            aggregation=aggregation_metadata,
            frame_results=frame_results_list,
            gradcam_video_url=None,  # Will be set by router
            processing_time_seconds=round(processing_time, 3)
        )

        self.logger.info(f"Analysis complete: {result.aggregate_prediction} "
                        f"(confidence={result.aggregate_confidence:.4f}, "
                        f"authenticity={authenticity_score:.1f}%, "
                        f"time={processing_time:.2f}s)")

        return response, gradcam_output_path if options.enable_gradcam else None

    def _calculate_authenticity_score(self, result: VideoInferenceResult) -> float:
        """Calculate authenticity score (0-100) from inference result.

        Score of 100 = definitely real, 0 = definitely fake.

        Parameters:
            result: Video inference result

        Returns:
            Authenticity score between 0 and 100
        """
        if result.aggregate_prediction == "REAL":
            return round(result.aggregate_confidence * 100, 2)
        else:
            return round((1 - result.aggregate_confidence) * 100, 2)

    def cleanup_file(self, file_path: Path) -> None:
        """Clean up a file after processing.

        Parameters:
            file_path: Path to file to delete
        """
        try:
            if file_path.exists():
                file_path.unlink()
                self.logger.debug(f"Cleaned up file: {file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup file {file_path}: {e}")

    def cleanup_directory(self, dir_path: Path) -> None:
        """Clean up a directory and its contents.

        Parameters:
            dir_path: Path to directory to delete
        """
        try:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                self.logger.debug(f"Cleaned up directory: {dir_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup directory {dir_path}: {e}")
