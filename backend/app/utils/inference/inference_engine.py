from logging import getLogger, Logger
import logging
import torch
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
import urllib.request
import cv2
import numpy as np
from core.model import DeepFakeDetector
from preprocessing.face_detector import FaceDetector
from preprocessing.face_extractor import FaceExtractor
from preprocessing.optimized_frames_extractor import create_frames_extractor
from torchvision import transforms
from utils.inference_results_dataclasses import InferenceResult, VideoInferenceResult
from inference.score_aggregator import ScoreAggregator, AggregatedScore
from inference.batch_prefetcher import BatchPrefetcher
from utils.config_dataclasses import InferenceConfig
from config import (
    PROJECT_ROOT,
    MODEL_FILENAME,
    FACE_DETECTION_MODEL_PATH,
    FACE_DETECTION_HF_URL,
    HF_MODEL_URL
)
from gradcam.gradcam import GradCAM, GradCAMVisualizer


class InferenceEngine:
    """Inference engine for VeridisQuo deepfake detection model."""

    CLASS_LABELS: List[str] = ["FAKE", "REAL"]
    """Supported class labels for model predictions (FAKE=0, REAL=1)"""
 
    def __init__(self, config: InferenceConfig, **kwargs):
        """Initialize InferenceEngine.
        Parameters:
            config (InferenceConfig): configuration for entire inference engine and model
            kwargs: override config parameters for critical settings, allowing to run without full config
                TODO
        Raises:
            ValueError: if config is not provided and no valid parameter is given for fallback
            RuntimeError: if model loading fails
        """
        super().__init__()

        self.logger: Logger = getLogger("/".join(__file__.split("/")[-2:]))
        """Logger instance for InferenceEngine"""

        if config is None:
            raise ValueError("Configuration is required. Please provide an InferenceConfig instance.")

        if not isinstance(config, InferenceConfig):
            raise TypeError(f"config must be InferenceConfig, got {type(config).__name__}")

        self.config: InferenceConfig = config

        #TODO: handle kwargs overrides for critical settings
        #Example: model_path, device, auto_download, etc.

        self.device: torch.device = self.config.device.get_device()
        """Device to run inference on (CPU or GPU)"""
        self.logger.info(f"Model will be loaded on device: {self.device} (cuda available: {torch.cuda.is_available()})")

        self.model_path: Path
        """Path to model weights file"""
        if self.config.paths.model_path is None:
            self.model_path = PROJECT_ROOT / "models" / "deepfake_detection" / MODEL_FILENAME
        else:
            self.model_path = Path(self.config.paths.model_path)

        if not self.model_path.exists() and self.config.inference_options.auto_download_model:
            self.logger.warning(f"Model not found at {self.model_path}, downloading from Hugging Face...")
            self._download_model()

        if not self.model_path.exists():
            error_msg = f"Model file not found at {self.model_path}. Set auto_download=True to download automatically."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Initializing VeridisQuo deepfake detection model...")

        self.model: DeepFakeDetector = DeepFakeDetector(config=self.config.model_architecture)
        """DeepFakeDetector model instance"""

        self._load_weights()
        self.logger.info("Model initialized and weights loaded successfully.")

        self.model = self.model.to(self.device)
        self.logger.info(f"Model moved to device: {self.device}")
        self.model.eval()
        self.logger.info("Model set to evaluation mode.")

        self.face_detector: Optional[FaceDetector] = None
        """FaceDetector instance for video inference, if available"""

        face_detection_model_path = Path(self.config.paths.face_detection_model_path) if self.config.paths.face_detection_model_path else FACE_DETECTION_MODEL_PATH

        if self.config.preprocessing.face_detection.enabled:
            # Auto-download face detection model if not found
            if not face_detection_model_path.exists() and self.config.inference_options.auto_download_model:
                self.logger.warning(f"Face detection model not found at {face_detection_model_path}, downloading from Hugging Face...")
                self._download_face_detection_model(face_detection_model_path)

            if face_detection_model_path.exists():
                self.logger.info("Initializing FaceDetector...")

                self.face_detector = FaceDetector(
                    model_path=str(face_detection_model_path),
                    device=str(self.device),
                    min_face_size=self.config.preprocessing.face_detection.min_face_size,
                    confidence_threshold=self.config.preprocessing.face_detection.confidence_threshold,
                    only_keep_top=self.config.preprocessing.face_detection.only_keep_top
                )
            else:
                self.logger.warning(
                    f"Face detection model not found at {face_detection_model_path} and auto-download is disabled. "
                    "Video inference will process entire frames instead of detected faces."
                )
        else:
            self.logger.info("Face detection is disabled in configuration.")

        self.image_size: Tuple[int, int] = self.config.preprocessing.face_extraction.target_size
        """Image size for preprocessing (width, height)"""

        self.mean: List[float] = self.config.preprocessing.imagenet_mean
        """ImageNet mean for normalization"""

        self.std: List[float] = self.config.preprocessing.imagenet_std
        """ImageNet std for normalization"""

        self.padding: int = self.config.preprocessing.face_extraction.padding
        """Padding around detected face bounding box"""

        self.use_face_detection: bool = self.face_detector is not None
        """Whether face detection is available for video inference"""
        self.logger.info(f"Face detection enabled: {self.use_face_detection}")

        self.face_extractor: FaceExtractor = FaceExtractor(
            target_size=self.image_size,
            normalization=self.config.preprocessing.face_extraction.normalization,
            padding=self.padding
        )
        """Face extractor for consistent preprocessing"""

        self.transform: transforms.Compose = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        """Torchvision transform pipeline for normalization"""

        self.gradcam: Optional[GradCAM] = None
        """GradCAM instance (initialized on-demand)"""

        self.gradcam_target_layer = None
        """Target conv layer for GradCAM visualization"""

        self.logger.info("InferenceEngine initialized successfully")

    def _init_gradcam(self) -> None:
        """Initialize GradCAM on-demand for video visualization.

        Uses the last convolutional layer of EfficientNet as target.
        This is called lazily when GradCAM is first requested.
        """
        if self.gradcam is None:
            self.gradcam_target_layer = self.model.efficient.features[-1]
            self.gradcam = GradCAM(self.model, self.gradcam_target_layer)
            self.logger.info("GradCAM initialized with target layer: efficient.features[-1]")

    def _download_model(self) -> None:
        """Download model weights from Hugging Face.
        Raises:
            RuntimeError: if download fails
        """
        try:
            self.logger.info(f"Downloading model from {HF_MODEL_URL}...")

            #create parent directory if needed
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

            def _progress_hook(count: int, block_size: int, total_size: int) -> None:
                """Progress callback for model download."""
                percent = int(count * block_size * 100 / total_size)
                if count % 100 == 0:
                    self.logger.info(f"Download progress: {percent}%")

            urllib.request.urlretrieve(
                HF_MODEL_URL,
                self.model_path,
                reporthook=_progress_hook
            )

            self.logger.info(f"Model downloaded successfully to {self.model_path}")

        except Exception as e:
            error_msg = f"Failed to download model from Hugging Face: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _download_face_detection_model(self, model_path: Path) -> None:
        """Download face detection model from Hugging Face.

        Parameters:
            model_path: Path where the model should be saved

        Raises:
            RuntimeError: if download fails
        """
        try:
            self.logger.info(f"Downloading face detection model from {FACE_DETECTION_HF_URL}...")

            # Create parent directory if needed
            model_path.parent.mkdir(parents=True, exist_ok=True)

            def _progress_hook(count: int, block_size: int, total_size: int) -> None:
                """Progress callback for model download."""
                percent = int(count * block_size * 100 / total_size)
                if count % 100 == 0:
                    self.logger.info(f"Download progress: {percent}%")

            urllib.request.urlretrieve(
                FACE_DETECTION_HF_URL,
                model_path,
                reporthook=_progress_hook
            )

            self.logger.info(f"Face detection model downloaded successfully to {model_path}")

        except Exception as e:
            error_msg = f"Failed to download face detection model from Hugging Face: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _load_weights(self) -> None:
        """Load pretrained weights into the model.
        Raises:
            RuntimeError: if weight loading fails
        """
        try:
            self.logger.info(f"Loading model weights from {self.model_path}...")

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # Handle both direct state_dict and training checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                self.logger.debug("Loaded model_state_dict from training checkpoint")
            else:
                state_dict = checkpoint
                self.logger.debug("Loaded direct state_dict")

            self.model.load_state_dict(state_dict)

            self.logger.info("Model weights loaded successfully")

        except Exception as e:
            error_msg = f"Failed to load model weights: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _extract_face(self,
        image: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Extract face region from image using detections.
        Parameters:
            image (np.ndarray): input image in BGR format [H, W, 3]
            detections (List[Dict]): face detections from face_detector
        Returns:
            np.ndarray: extracted and preprocessed face region [224, 224, 3]
        """
        if detections:
            #use the top face (highest confidence)
            top_face = detections[0]
            bbox = top_face['bbox']

            #extract face region with padding
            face_region, _ = self.face_extractor.extract_and_preprocess(
                image, bbox, normalize=False
            )
        else:
            #no face detected, use entire frame
            face_region = self.face_extractor.preprocess_face(image, normalize=False)

        return face_region

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single image (wrapper around batch method).
        Parameters:
            image (np.ndarray): Input image in BGR format [H, W, 3]
        Returns:
            torch.Tensor: Preprocessed image tensor [1, 3, 224, 224]
        """
        return self._preprocess_image_batch([image])  #[1, 3, 224, 224]

    def _preprocess_image_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess a batch of images efficiently with batched operations.
        Parameters:
            images (List[np.ndarray]): list of images in BGR format [H, W, 3]
        Returns:
            torch.Tensor: batched preprocessed tensor [N, 3, 224, 224]
        Raises:
            RuntimeError: if all images fail preprocessing
        """
        #batch face detection wrapped into a single YOLO forward pass
        if self.use_face_detection:
            detections_batch = self.face_detector.detect_faces_batch(images)
        else:
            detections_batch = [[] for _ in images]

        face_regions = []
        for idx, (image, detections) in enumerate(zip(images, detections_batch)):
            try:
                face_region = self._extract_face(image, detections)
                face_regions.append(face_region)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract face from image {idx}/{len(images)}: {e}, skipping frame")
                continue

        if not face_regions:
            raise RuntimeError("All images in batch failed face extraction")

        #batch transformations (BGR→RGB + ToTensor + Normalize)
        face_rgbs = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in face_regions]
        tensors = [self.transform(face_rgb) for face_rgb in face_rgbs]

        #batch tensor [N, 3, 224, 224]
        batch_tensor = torch.stack(tensors, dim=0)

        return batch_tensor

    def _run_inference(self, preprocessed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run model inference on preprocessed tensor.
        Parameters:
            preprocessed (torch.Tensor): Preprocessed image tensor [N, 3, 224, 224]
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (logits, probabilities, prediction) all on GPU
        """
        with torch.no_grad():
            logits = self.model(preprocessed) #[N, 2]
            prediction = self.model.classifier.predict(logits) #[N]
            probabilities = self.model.classifier.predict_proba(logits) #[N, 2]
        return logits, probabilities, prediction

    def _predict_frame_batch(self, frames: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a batch of frames with batched operations.
        Parameters:
            frames (List[np.ndarray]): List of frame images in BGR format [H, W, 3]
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: (logits, probabilities, prediction, preprocessed_tensor)
                - logits: [N, 2] on GPU
                - probabilities: [N, 2] on GPU
                - prediction: [N] on GPU (class indices)
                - preprocessed_tensor: [N, 3, 224, 224] on GPU (for GradCAM)
        Raises:
            ValueError: if frames list is empty
            RuntimeError: if all frames fail preprocessing
        """
        if not frames:
            raise ValueError("frames list cannot be empty")

        #batch preprocessing (includes batched face detection + transforms)
        batch_tensor = self._preprocess_image_batch(frames)  # [N, 3, 224, 224] CPU

        #single GPU transfer for entire batch
        batch_tensor = batch_tensor.to(self.device)

        logits, probabilities, prediction = self._run_inference(batch_tensor)

        return logits, probabilities, prediction, batch_tensor

    def _validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate that file (video or image) path exists.
        Parameters:
            file_path (Union[str, Path]): Path to video file
        Returns:
            Path: Validated path object
        Raises:
            FileNotFoundError: if file doesn't exist
        """
        file_path = Path(file_path)
        try:
            assert file_path.exists(), f"File not found: {file_path}"
            
        except AssertionError as e:
            error_msg = f"Invalid parameters: {e}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        return file_path

    def _build_frame_result(self,
        predicted_class: int,
        probs_cpu: np.ndarray,
        logits_cpu: np.ndarray,
        frame_idx: int
    ) -> InferenceResult:
        """Build InferenceResult for a single frame from batch predictions.
        Parameters:
            predicted_class (int): Predicted class index
            probs_cpu (np.ndarray): Probabilities array [N, 2]
            logits_cpu (np.ndarray): Logits array [N, 2]
            frame_idx (int): Index of frame in batch
        Returns:
            InferenceResult: Result object for the frame
        """
        confidence = float(probs_cpu[frame_idx, predicted_class])
        prediction_label = self.CLASS_LABELS[predicted_class]

        prob_dict = {
            label: float(probs_cpu[frame_idx, j])
            for j, label in enumerate(self.CLASS_LABELS)
        }

        return InferenceResult(
            prediction=prediction_label,
            confidence=confidence,
            probabilities=prob_dict,
            raw_logits=logits_cpu[frame_idx].tolist()
        )

    def _process_video_batches(
        self,
        prefetcher: BatchPrefetcher,
        enable_gradcam: bool = False
    ) -> Tuple[List[InferenceResult], List[np.ndarray], List[Dict], List[Dict]]:
        """Process all batches from prefetcher and return frame results.
        Parameters:
            prefetcher (BatchPrefetcher): Initialized prefetcher with batches ready
            enable_gradcam (bool): Whether to generate GradCAM heatmaps
        Returns:
            Tuple[List[InferenceResult], List[np.ndarray], List[Dict], List[Dict]]:
                - frame_results: inference results per frame
                - gradcam_heatmaps: raw heatmaps [224, 224] in [0, 1]
                - face_infos: bbox and shape info for each heatmap
                - frame_metadata: frame numbers and timestamps
        Raises:
            RuntimeError: if no frames were successfully processed
        """
        frame_results: List[InferenceResult] = []
        gradcam_heatmaps: List[np.ndarray] = []
        face_infos: List[Dict] = []
        all_frame_metadata: List[Dict] = []

        try:
            for batch_frames, batch_metadata in prefetcher:
                try:
                    logits, probabilities, predictions, preprocessed = self._predict_frame_batch(batch_frames)

                    #single CPU transfer per batch
                    probs_cpu = probabilities.cpu().numpy()  # [N, 2]
                    logits_cpu = logits.cpu().numpy()  # [N, 2]
                    predictions_cpu = predictions.cpu().numpy()  # [N]

                    # Generate GradCAM heatmaps if enabled
                    if enable_gradcam:
                        heatmaps, face_info_batch = self._generate_gradcam_heatmaps_batch(
                            batch_frames=batch_frames,
                            preprocessed_tensor=preprocessed,
                            target_class=0  # FAKE class
                        )
                        gradcam_heatmaps.extend(heatmaps)
                        face_infos.extend(face_info_batch)
                        all_frame_metadata.extend(batch_metadata)

                    #build results for each frame in batch
                    for i in range(len(batch_frames)):
                        predicted_class = int(predictions_cpu[i])
                        result = self._build_frame_result(predicted_class, probs_cpu, logits_cpu, i)
                        frame_results.append(result)

                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(
                                f"Frame {batch_metadata[i]['frame_number']} @ "
                                f"{batch_metadata[i]['timestamp']:.2f}s: "
                                f"{result.prediction} ({result.confidence:.4f})"
                            )

                except Exception as e:
                    self.logger.warning(f"Failed to process batch of {len(batch_frames)} frames: {e}")
                    # Skip GradCAM for failed batches
                    if enable_gradcam:
                        # Add empty heatmaps to keep sync
                        for _ in range(len(batch_frames)):
                            gradcam_heatmaps.append(None)
                            face_infos.append(None)
                        all_frame_metadata.extend(batch_metadata)

        finally:
            prefetcher.stop()

        if not frame_results:
            raise RuntimeError("No frames were successfully processed")

        return frame_results, gradcam_heatmaps, face_infos, all_frame_metadata

    def _aggregate_video_results(self,
        frame_results: List[InferenceResult],
        video_path: Path,
        aggregation_method: str
    ) -> VideoInferenceResult:
        """Aggregate frame results into video-level prediction.
        Parameters:
            frame_results (List[InferenceResult]): Results for all frames
            video_path (Path): Path to video file
            aggregation_method (str): Aggregation method to use
        Returns:
            VideoInferenceResult: Aggregated video prediction
        """
        self.logger.info(f"Processed {len(frame_results)} frames from video")

        aggregator = ScoreAggregator(method=aggregation_method)
        aggregated = aggregator.aggregate(frame_results)

        video_result = VideoInferenceResult(
            video_path=str(video_path),
            frame_results=frame_results,
            aggregate_prediction=aggregated.prediction,
            aggregate_confidence=aggregated.confidence,
            num_frames_analyzed=len(frame_results),
            aggregation_metadata={
                "method": aggregated.method,
                "frame_count": aggregated.frame_count,
                **aggregated.metadata
            }
        )

        self.logger.info(
            f"Video prediction: {aggregated.prediction} "
            f"(confidence: {aggregated.confidence:.4f}, "
            f"method: {aggregation_method}, "
            f"frames: {len(frame_results)})"
        )

        return video_result

    def _generate_gradcam_heatmaps_batch(
        self,
        batch_frames: List[np.ndarray],
        preprocessed_tensor: torch.Tensor,
        target_class: int = 0
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """Generate raw GradCAM heatmaps for a batch of frames.

        Parameters:
            batch_frames: Original frames in BGR format [H, W, 3]
            preprocessed_tensor: Preprocessed tensor [N, 3, 224, 224] (normalized, RGB)
            target_class: Class to visualize (0=FAKE, 1=REAL)

        Returns:
            Tuple[List[np.ndarray], List[Dict]]:
                - heatmaps: [N] list of [224, 224] float32 arrays in [0, 1]
                - face_infos: [N] list of dicts with 'bbox' and 'frame_shape'
        """
        try:
            # Generate CAM heatmaps for entire batch
            cams = self.gradcam(
                preprocessed_tensor,
                target_class=target_class,
                upsample_to=(224, 224)
            )  # [N, 224, 224] normalized to [0, 1]

            heatmaps = []
            face_infos = []

            for i, (frame, cam) in enumerate(zip(batch_frames, cams)):
                cam_np = cam.cpu().numpy()  # [224, 224] float32 [0, 1]

                # Get face bounding box
                if self.use_face_detection:
                    detections = self.face_detector.detect_faces_batch([frame])[0]
                    if detections:
                        detection = detections[0]
                        x, y, width, height = detection['bbox']
                        bbox = (x, y, x + width, y + height)  # (x1, y1, x2, y2)
                    else:
                        bbox = None
                        self.logger.warning(f"No face detected in frame {i}")
                else:
                    bbox = None

                face_info = {
                    'bbox': bbox,
                    'frame_shape': frame.shape[:2]  # (H, W)
                }

                heatmaps.append(cam_np)
                face_infos.append(face_info)

            return heatmaps, face_infos

        except Exception as e:
            self.logger.error(f"GradCAM heatmap generation failed: {e}")
            # Return empty lists on error
            return [], []

    def _apply_heatmap_to_frame(
        self,
        frame: np.ndarray,
        heatmap: np.ndarray,
        face_info: Dict,
        alpha: float
    ) -> np.ndarray:
        """Apply GradCAM heatmap overlay to a single frame.

        Parameters:
            frame: Original frame [H, W, 3] BGR
            heatmap: Raw heatmap [224, 224] float32 in [0, 1]
            face_info: Dict with 'bbox' (x1, y1, x2, y2) and 'frame_shape'
            alpha: Overlay transparency

        Returns:
            Frame with heatmap overlay [H, W, 3] BGR
        """
        frame_with_overlay = frame.copy()
        bbox = face_info.get('bbox') if face_info else None

        if bbox is None:
            # No face detected - return original frame
            return frame

        x1, y1, x2, y2 = bbox

        # Extract face region
        face_region = frame[y1:y2, x1:x2]
        face_h, face_w = face_region.shape[:2]

        if face_h == 0 or face_w == 0:
            return frame

        # Resize heatmap to face size
        heatmap_resized = cv2.resize(heatmap, (face_w, face_h))

        # Convert to RGB for colormap application
        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

        # Apply colormap to heatmap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = GradCAMVisualizer.apply_colormap(heatmap_uint8)

        # Create elliptical mask to match face shape
        mask = np.zeros((face_h, face_w), dtype=np.uint8)
        center = (face_w // 2, face_h // 2)
        # Use 90% of bbox dimensions for ellipse axes to stay within face
        axes = (int(face_w * 0.45), int(face_h * 0.45))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # Apply feathering (Gaussian blur) to soften edges
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)

        # Overlay heatmap on face with mask
        overlay_rgb = cv2.addWeighted(
            face_rgb, 1 - alpha,
            heatmap_colored, alpha,
            0
        )

        # Blend overlay with original face using elliptical mask
        face_rgb_blended = (
            face_rgb * (1 - mask_3ch) +
            overlay_rgb * mask_3ch
        ).astype(np.uint8)

        # Convert back to BGR
        overlay_bgr = cv2.cvtColor(face_rgb_blended, cv2.COLOR_RGB2BGR)

        # Place overlay back on frame
        frame_with_overlay[y1:y2, x1:x2] = overlay_bgr

        return frame_with_overlay

    def _save_gradcam_video(
        self,
        gradcam_heatmaps: List[np.ndarray],
        face_infos: List[Dict],
        frame_metadata: List[Dict],
        output_path: Path,
        original_video_path: Path,
        fps: float,
        alpha: float = 0.4
    ) -> None:
        """Save original video with interpolated GradCAM overlays.

        Reads all frames from original video and applies interpolated GradCAM heatmaps.
        Only the heatmaps are interpolated, not the video frames themselves.

        Parameters:
            gradcam_heatmaps: List of raw heatmaps [224, 224] float32 in [0, 1]
            face_infos: List of dicts with 'bbox' and 'frame_shape' for each heatmap
            frame_metadata: Metadata for each sampled frame (includes frame_number)
            output_path: Path to save video
            original_video_path: Path to original video to read all frames from
            fps: Original video FPS
            alpha: Overlay transparency (0-1)

        Raises:
            RuntimeError: if video writing fails
        """
        try:
            if not gradcam_heatmaps or not original_video_path:
                self.logger.warning("No heatmaps or original video path provided")
                return

            # Open original video
            cap = cv2.VideoCapture(str(original_video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open original video: {original_video_path}")

            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # Initialize VideoWriter with H.264 codec (better compression)
            # Try H.264/AVC first for better compression
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )

            if not out.isOpened():
                # Fallback to mp4v if H.264 is not available
                self.logger.warning("H.264 codec (avc1) not available, falling back to mp4v")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (width, height)
                )

                if not out.isOpened():
                    cap.release()
                    raise RuntimeError(f"Failed to open VideoWriter for {output_path} with both avc1 and mp4v codecs")

            # Get frame numbers from metadata
            frame_numbers = [meta['frame_number'] for meta in frame_metadata]

            self.logger.info(
                f"Applying GradCAM overlays: {len(gradcam_heatmaps)} heatmaps for {total_frames} total frames"
            )

            # Process each frame from original video
            frame_idx = 0
            while True:
                ret, original_frame = cap.read()
                if not ret:
                    break

                # Get or interpolate heatmap for this frame
                if frame_idx in frame_numbers:
                    # Exact match - use computed heatmap
                    heatmap_idx = frame_numbers.index(frame_idx)
                    heatmap = gradcam_heatmaps[heatmap_idx]
                    face_info = face_infos[heatmap_idx]
                else:
                    # Interpolate heatmap
                    prev_idx = None
                    next_idx = None

                    for idx, fnum in enumerate(frame_numbers):
                        if fnum < frame_idx:
                            prev_idx = idx
                        elif fnum > frame_idx and next_idx is None:
                            next_idx = idx
                            break

                    if prev_idx is not None and next_idx is not None:
                        # Linear interpolation of heatmaps
                        prev_frame_num = frame_numbers[prev_idx]
                        next_frame_num = frame_numbers[next_idx]
                        weight = (frame_idx - prev_frame_num) / (next_frame_num - prev_frame_num)

                        # Interpolate heatmaps
                        if gradcam_heatmaps[prev_idx] is not None and gradcam_heatmaps[next_idx] is not None:
                            heatmap = cv2.addWeighted(
                                gradcam_heatmaps[prev_idx], 1 - weight,
                                gradcam_heatmaps[next_idx], weight,
                                0
                            )
                            # Interpolate bboxes to prevent teleportation
                            prev_bbox = face_infos[prev_idx].get('bbox')
                            next_bbox = face_infos[next_idx].get('bbox')

                            if prev_bbox is not None and next_bbox is not None:
                                # Linear interpolation of bbox coordinates
                                interpolated_bbox = tuple(
                                    int(prev_bbox[i] * (1 - weight) + next_bbox[i] * weight)
                                    for i in range(4)
                                )
                                face_info = {
                                    'bbox': interpolated_bbox,
                                    'frame_shape': face_infos[prev_idx]['frame_shape']
                                }
                            else:
                                # One bbox is None, use the valid one
                                face_info = face_infos[prev_idx] if prev_bbox else face_infos[next_idx]
                        else:
                            # One of the heatmaps is None, use the valid one
                            heatmap = gradcam_heatmaps[prev_idx] or gradcam_heatmaps[next_idx]
                            face_info = face_infos[prev_idx] or face_infos[next_idx]
                    elif prev_idx is not None:
                        heatmap = gradcam_heatmaps[prev_idx]
                        face_info = face_infos[prev_idx]
                    elif next_idx is not None:
                        heatmap = gradcam_heatmaps[next_idx]
                        face_info = face_infos[next_idx]
                    else:
                        # No heatmaps available - write original frame
                        out.write(original_frame)
                        frame_idx += 1
                        continue

                # Apply heatmap to original frame
                if heatmap is not None and face_info is not None:
                    frame_with_overlay = self._apply_heatmap_to_frame(
                        original_frame,
                        heatmap,
                        face_info,
                        alpha
                    )
                else:
                    frame_with_overlay = original_frame

                out.write(frame_with_overlay)
                frame_idx += 1

            cap.release()
            out.release()
            self.logger.info(f"GradCAM video saved: {total_frames} frames written to {output_path}")

        except Exception as e:
            error_msg = f"Failed to save GradCAM video: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def predict_image(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference on a single image.
        Parameters:
            image (np.ndarray): Image array [H, W, 3] in BGR format
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (logits, probabilities, prediction)
                - logits: [1, 2] on GPU
                - probabilities: [1, 2] on GPU
                - prediction: [1] on GPU (class index)
        Raises:
            ValueError: if image is invalid
            RuntimeError: if inference fails
        """
        try:
            assert image is not None and image.size != 0, "Image is empty or None"

            preprocessed = self._preprocess_image(image)  #[1, 3, 224, 224]
            preprocessed = preprocessed.to(self.device)
            logits, probabilities, prediction = self._run_inference(preprocessed)
            return logits, probabilities, prediction

        except AssertionError as e:
            error_msg = f"Invalid parameter: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Inference failed for image array: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def predict_image(self, image: Union[str, Path]) -> InferenceResult:
        """Run inference on a single image.
        Parameters:
            image (str or Path): Path to the image file
        Returns:
            InferenceResult: Prediction result with confidence and probabilities.
        Raises:
            FileNotFoundError: if image file doesn't exist
            RuntimeError: if inference fails
        """
        try:
            image_path = self._validate_file_path(image)
            image_array = cv2.imread(str(image_path))
            assert image_array is not None, f"Failed to load image: {image_path}"

            self.logger.info(f"Processing image: {image_path.name}")

            #reuse ndarray implementation
            logits, probabilities, prediction = self.predict_image(image_array)  # [1, 2], [1, 2], [1] on GPU

            #CPU transfer and result construction
            probs_cpu = probabilities.cpu().numpy()[0]  # [2]
            logits_cpu = logits.cpu().numpy()[0]  # [2]
            predicted_class = int(prediction.cpu().item())  # Use prediction instead of argmax

            confidence = float(probs_cpu[predicted_class])
            prediction_label = self.CLASS_LABELS[predicted_class]

            prob_dict = {
                label: float(prob)
                for label, prob in zip(self.CLASS_LABELS, probs_cpu)
            }

            result = InferenceResult(
                prediction=prediction_label,
                confidence=confidence,
                probabilities=prob_dict,
                raw_logits=logits_cpu.tolist()
            )

            self.logger.info(f"Predicted image: {prediction_label} (confidence: {confidence:.4f})")

            return result
        
        except Exception as e:
            error_msg = f"Inference failed for image array: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
    def predict_video(self,
        video_path: Union[str, Path],
        frames_per_second: Optional[int] =1,
        aggregation_method: str ="majority",
        use_pyav_extractor: bool =True,
        batch_size: int =8,
        enable_gradcam: bool = False,
        gradcam_output_path: Optional[Union[str, Path]] = None,
        gradcam_alpha: float = 0.4
    ) -> VideoInferenceResult:
        """Run inference on a video file.
        Parameters:
            video_path (str or Path): Path to the video file
            frames_per_second (int, optional): Number of frames to sample per second.
                If None, processes all frames (can be slow).
            aggregation_method (str): Method to aggregate frame predictions.
                Options: "majority", "average", "weighted_average", "max_confidence", "threshold"
            use_pyav_extractor (bool): Whether to use OptimizedFramesExtractor (PyAV)
            batch_size (int): Number of frames to process in a single batch (default: 8)
            enable_gradcam (bool): If True, generate GradCAM visualization video
            gradcam_output_path (str or Path, optional): Path to save GradCAM video (required if enable_gradcam=True)
            gradcam_alpha (float): Transparency for heatmap overlay (0=transparent, 1=opaque), default 0.4
        Returns:
            VideoInferenceResult: Aggregated video prediction result
        Raises:
            FileNotFoundError: if video file doesn't exist
            ValueError: if enable_gradcam=True but gradcam_output_path is None
            RuntimeError: if inference fails
        """
        video_path = self._validate_file_path(video_path)
        self.logger.info(f"Processing video: {video_path.name}")

        # Validate GradCAM parameters
        if enable_gradcam and gradcam_output_path is None:
            raise ValueError("gradcam_output_path must be specified when enable_gradcam=True")

        if enable_gradcam:
            gradcam_output_path = Path(gradcam_output_path)
            gradcam_output_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_gradcam()  # Lazy initialization

        try:
            #create frames extractor
            frames_extractor = create_frames_extractor(
                video_path=str(video_path),
                nb_fps=frames_per_second,
                device=str(self.device),
                use_optimized=use_pyav_extractor
            )

            self.logger.info(f"Extracting frames at {frames_per_second} FPS (video FPS: {frames_extractor.video_fps:.1f})")

            #create prefetcher for async batch preparation
            prefetcher = BatchPrefetcher(
                frames_extractor.extract_frames(),
                batch_size=batch_size,
                prefetch_size=2
            )
            prefetcher.start()

            # Process video with optional GradCAM
            frame_results, gradcam_heatmaps, face_infos, frame_metadata = self._process_video_batches(
                prefetcher,
                enable_gradcam=enable_gradcam
            )

            # Save GradCAM video if enabled
            if enable_gradcam and gradcam_heatmaps:
                self._save_gradcam_video(
                    gradcam_heatmaps=gradcam_heatmaps,
                    face_infos=face_infos,
                    frame_metadata=frame_metadata,
                    output_path=gradcam_output_path,
                    original_video_path=video_path,
                    fps=frames_extractor.video_fps,
                    alpha=gradcam_alpha
                )

            # Aggregate and build final result
            video_result = self._aggregate_video_results(frame_results, video_path, aggregation_method)

            # Add GradCAM path to result if generated
            if enable_gradcam:
                video_result.gradcam_output_path = str(gradcam_output_path)

            return video_result

        except Exception as e:
            error_msg = f"Video inference failed for {video_path}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def predict_video_batch(
        self,
        video_paths: List[Union[str, Path]],
        frames_per_second: Optional[int] = 1,
        aggregation_method: str = "majority",
        use_pyav_extractor: bool = True,
        batch_size: int = 8
    ) -> List[VideoInferenceResult]:
        """Run inference on a batch of videos.
        Parameters:
            video_paths (List[str or Path]): List of video file paths
            frames_per_second (int, optional): Number of frames to sample per second
            aggregation_method (str): Method to aggregate frame predictions
            use_pyav_extractor (bool): Whether to use OptimizedFramesExtractor
            batch_size (int): Number of frames to process in a single batch (default: 8)
        Returns:
            List[VideoInferenceResult]: List of video prediction results
        Raises:
            ValueError: if video_paths is empty
        """
        try:
            assert video_paths, "List of video paths cannot be empty"
            
        except AssertionError as e:
            error_msg = f"Invalid parameter: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(f"Processing batch of {len(video_paths)} videos")

        results: List[VideoInferenceResult] = []

        for video_path in video_paths:
            try:
                result = self.predict_video(
                    video_path=video_path,
                    frames_per_second=frames_per_second,
                    aggregation_method=aggregation_method,
                    use_pyav_extractor=use_pyav_extractor,
                    batch_size=batch_size
                )
                results.append(result)

            except FileNotFoundError as e:
                self.logger.warning(f"Video file not found: {video_path}, skipping.")      
            
            except Exception as e:
                self.logger.error(f"Failed to process video {video_path}: {e}")
                
            finally:  
                #create error result
                error_result = VideoInferenceResult(
                    video_path=str(video_path),
                    frame_results=[],
                    aggregate_prediction="Inference Failed",
                    aggregate_confidence=0.0,
                    num_frames_analyzed=0,
                    aggregation_metadata={"error": str(e)}
                )
                results.append(error_result)
                continue

        self.logger.info(f"Batch video inference complete: {len(results)} results")
        return results