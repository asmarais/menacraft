import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Union
import logging
from ultralytics import YOLO
from pathlib import Path
from config import FACE_DETECTION_MODEL_PATH


class FaceDetector:
    """Detect faces in images using YOLO."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_face_size: int = 40,
        confidence_threshold: float = 0.7,
        only_keep_top: bool = True,
        device: Union[str, int] = "cpu"
    ):
        """Initializes the YOLO face detector.

        Parameters:
            model_path (Optional[str]): path to the YOLO face detection model.
                                       If None, uses the default path from config.
            min_face_size (int): minimum face size (in pixels) - faces smaller than this will be filtered out
            confidence_threshold (float): minimum confidence score to accept detection
            only_keep_top (bool): if True and multiple faces detected, keep only the highest confidence one
            device (Union[str, int]): device to run inference on. Can be 'cpu', 'cuda', or GPU index (0, 1, etc.)

        Raises:
            Exception: if YOLO model initialization fails or model file not found
        """
        self.min_face_size: int = min_face_size
        """Minimum face size (in pixels)"""
        self.confidence_threshold: float = confidence_threshold
        """Minimum confidence score to accept a detection"""
        self.only_keep_top: bool = only_keep_top
        """If True and multiple faces detected, keep only the highest confidence one"""
        self.device: Union[str, int] = device
        """Device to run inference on (cpu, cuda, or GPU index)"""
        self.logger: logging.Logger = logging.getLogger("preprocessing.FaceDetector")
        """Logger for the FaceDetector class"""

        try:
            # Use default path from config if none provided
            if model_path is None:
                model_path = str(FACE_DETECTION_MODEL_PATH)

            # Check if model file exists
            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"YOLO model not found at: {model_path}")

            # Load YOLO model
            self.detector: YOLO = YOLO(model_path)
            """YOLO face detector instance"""

            # Move model to specified device
            self.detector.to(device)

            self.logger.info(f"YOLO face detector initialized successfully from {model_path} on device: {device}")

        except Exception as e:
            self.logger.fatal(f"Failed to initialize YOLO face detector: {e}")
            raise

    def _process_single_result(
        self,
        result,
        return_landmarks: bool = False
    ) -> List[Dict[str, Any]]:
        """Process a single YOLO result object into detection dictionaries.
        Parameters:
            result: YOLO result object for one image
            return_landmarks (bool): whether to include facial landmarks
        Returns:
            List[Dict[str, Any]]: list of detection dictionaries for this image
        """
        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections = []

        #process each detection
        for box in result.boxes:
            confidence = float(box.conf[0])

            #filter by confidence threshold
            if confidence < self.confidence_threshold:
                continue

            #get bounding box in xyxy format and convert to xywh
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            x, y = int(x1), int(y1)
            width, height = int(x2 - x1), int(y2 - y1)

            #filter by minimum face size
            if width < self.min_face_size or height < self.min_face_size:
                continue

            detection = {
                'bbox': [x, y, width, height],
                'confidence': confidence
            }

            if return_landmarks:
                detection['keypoints'] = {}

            detections.append(detection)

        if not detections:
            return []

        #sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        #keep only top detection if requested
        if len(detections) > 1 and self.only_keep_top:
            detections = [detections[0]]

        return detections

    def detect_faces_batch(
        self,
        images: List[np.ndarray],
        return_landmarks: bool = False
    ) -> List[List[Dict[str, Any]]]:
        """Detect faces in multiple images with a single YOLO forward pass (batched).
        Parameters:
            images (List[np.ndarray]): list of images as numpy arrays (BGR format)
            return_landmarks (bool): whether to include facial landmarks in results
        Returns:
            List[List[Dict[str, Any]]]: list of detection lists (one per image)
        Raises:
            ValueError: if images list is empty or contains invalid images
        """
        if not images:
            raise ValueError("images list cannot be empty")

        try:
            #run YOLO inference on batch (single forward pass!)
            results = self.detector(images, verbose=False, device=self.device)

            if not results or len(results) == 0:
                self.logger.debug("No results from YOLO batch inference")
                return [[] for _ in images]

            #process each result
            all_detections = []
            for idx, result in enumerate(results):
                detections = self._process_single_result(result, return_landmarks)
                all_detections.append(detections)

                if detections:
                    self.logger.debug(f"Image {idx}: detected {len(detections)} face(s)")

            return all_detections

        except Exception as e:
            self.logger.error(f"Batch face detection failed: {e}")
            return [[] for _ in images]

    def detect_faces(
        self,
        image: np.ndarray,
        return_landmarks: bool = False
    ) -> List[Dict[str, Any]]:
        """Detect faces in an image using YOLO.

        Parameters:
            image (np.ndarray): input image as numpy array (BGR format)
            return_landmarks (bool): whether to include facial landmarks in results (not supported with YOLO)

        Returns:
            List[Dict[str, Any]]: list of detection dictionaries, each containing:
            - bbox: bounding box as [x, y, width, height]
            - confidence: detection confidence score
            - keypoints: facial landmarks (empty dict if return_landmarks=True, as YOLO doesn't provide landmarks)

        Raises:
            ValueError: if image is invalid
        """
        #validate input image
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            self.logger.error("Invalid image provided for face detection")
            raise ValueError("Invalid image: must be a non-empty numpy array")

        try:
            #run YOLO inference
            results = self.detector(image, verbose=False, device=self.device)

            if not results or len(results) == 0:
                self.logger.debug("No faces detected by YOLO")
                return []

            #process first result (single image) using helper method
            detections = self._process_single_result(results[0], return_landmarks)

            if detections:
                self.logger.debug(f"Detected {len(detections)} face(s)")
            else:
                self.logger.debug(
                    f"No faces with confidence >= {self.confidence_threshold} "
                    f"and size >= {self.min_face_size}px"
                )

            return detections

        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []

    def detect_single_face(
        self,
        image: np.ndarray,
        return_landmarks: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Detect a single face in an image (returns the one with the highest confidence score).

        Parameters:
            image (np.ndarray): input image as numpy array
            return_landmarks (bool): whether to include facial landmarks in result (not supported with YOLO)

        Returns:
            Optional[Dict[str, Any]]: None if no face found or dictionary containing:
            - bbox: bounding box as [x, y, width, height]
            - confidence: detection confidence score
            - keypoints: facial landmarks (empty dict if return_landmarks=True)

        Raises:
            ValueError: if provided image is invalid
        """
        detections = self.detect_faces(image, return_landmarks=return_landmarks)
        return None if not detections else detections[0]
