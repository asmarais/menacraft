import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
from PIL import Image

class FaceExtractor:
    """Extract and preprocess detected faces, handling resizing, normalization, and saving.
    Formats those features to be suitable for EfficientNet input."""

    AVAILABLE_NORMALIZATIONS_METHODS = ["zero_one", "minus_one_one", None]
    
    def __init__(self,
        target_size: Tuple[int, int] =(224, 224),
        normalization: str ="zero_one",
        padding: int = 0
    ):
        """Initialize FaceExtractor
        Parameters:
            target_size (Tuple[int, int]): target size for face images as (width, height)
            normalization (str): normalization method
            - "zero_one": scale pixels to [0, 1]
            - "minus_one_one": scale pixels to [-1, 1]
            - None: no normalization
            padding (int): number of pixels to add as padding around bbox
        Raises:
            ValueError: if parameters are invalid
        """
        self.logger = logging.getLogger("preprocessing.FaceExtractor")
        """Logger instance for the FaceExtractor class"""

        try:
            assert padding >= 0, "Padding must be >= 0"

        except AssertionError as e:
            self.logger.fatal(f"Invalid parameters: {e}.")
            raise ValueError(f"Invalid parameters: {e}.")
        
        if not (normalization in self.AVAILABLE_NORMALIZATIONS_METHODS):
            self.logger.warning(
                f"Invalid normalization method: {normalization}. Available methods are: "
                f"{self.AVAILABLE_NORMALIZATIONS_METHODS}. Falling back to 'zero_one'."
            )
            normalization = "zero_one"
        
        self.target_size = target_size
        """Target size for face images (width, height)"""
        self.normalization = normalization
        """Normalization method ('zero_one', 'minus_one_one', 'none')"""
        self.padding = padding
        """Number of pixels to add as padding around bounding box of the face"""
        
        self.logger.info("FaceExtractor initialized.")

    def preprocess_face(self,
        face_image: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """Preprocess a single face image, without padding.
        Parameters:
            face_image (np.ndarray): face image as numpy array (BGR format)
            normalize (bool): whether to apply normalization
        Returns:
            np.ndarray: preprocessed face image as numpy array
        Raises:
            ValueError: if face_image is invalid
        """
        if face_image is None or face_image.size == 0:
            self.logger.error(f"- preprocess_face - invalid face image: {face_image}")
            raise ValueError(f"Invalid face image: {face_image}")

        resized = cv2.resize(
            face_image,
            self.target_size,
            interpolation=cv2.INTER_AREA
        )

        return self._normalize(resized) if normalize else resized

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values
        Parameters:
            image (np.ndarray): input image
        Returns:
            np.ndarray: normalized image
        """
        match self.normalization:
            case "zero_one": 
                return image.astype(np.float32) / 255.0
            case "minus_one_one":
                return (image.astype(np.float32) / 127.5) - 1.0
            case _:
                return image

    def _denormalize(self, image: np.ndarray) -> np.ndarray:
        """Reverse normalization to get original pixel values
        Parameters:
            image: Normalized image
        Returns:
            np.ndarray: denormalized image with pixel values in [0, 255]
        """
        match self.normalization:
            case "zero_one":
                return (image * 255.0).clip(0, 255).astype(np.uint8)
            case "minus_one_one":
                return ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            case _:
                return image.astype(np.uint8)

    def extract_and_preprocess(self,
        frame: np.ndarray,
        bbox: list,
        normalize: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract face region from frame and preprocess it.
        Parameters:
            frame (np.ndarray): full frame image
            bbox (list): bounding box as [x, y, width, height]
            normalize (bool): whether to apply normalization
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (preprocessed_face, metadata)
        """
        face_region = self.extract_region(frame, bbox)
        preprocessed = self.preprocess_face(face_region, normalize)

        metadata = {
            "original_bbox": bbox,
            "original_size": face_region.shape[:2],
            "target_size": self.target_size,
            "normalization": self.normalization,
            "padding": self.padding
        }

        return preprocessed, metadata

    def extract_region(self,
        image: np.ndarray,
        bbox: List[int],
    ) -> np.ndarray:
        """Extract face region with padding from image using bounding box.
        Parameters:
            image (np.ndarray): input image
            bbox (List[int]): bounding box as [x, y, width, height]
        Returns:
            np.ndarray: extracted face region
        Raises:
            ValueError: if parameters are invalid
        """
        try:
            assert len(bbox) == 4, "Bounding box must be of the form [x, y, width, height]"

        except AssertionError as e:
            self.logger.error(f"- extract_region - {e}")
            raise ValueError(f"Invalid parameters for extract_region: {e}")

        x, y, w, h = bbox

        # Apply padding (in pixels)
        if self.padding > 0:
            x = max(0, x - self.padding)
            y = max(0, y - self.padding)
            w = min(image.shape[1] - x, w + self.padding * 2)
            h = min(image.shape[0] - y, h + self.padding * 2)

        # Ensure coordinates are within image bounds
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        x2 = min(x + w, image.shape[1])
        y2 = min(y + h, image.shape[0])

        return image[y:y2, x:x2]

    def save(self,
        face_image: np.ndarray,
        output_path: str,
        denormalize: bool = True
    ) -> None:
        """Save preprocessed face to disk as PNG.
        Parameters:
            face_image (np.ndarray): face image to save
            output_path (str): output file path
            denormalize_first (bool): whether to denormalize before saving
        Raises:
            ValueError: if parameters are invalid
        """
        try:
            assert face_image is not None and face_image.size > 0, "Invalid face image to save"
            assert Path.exists(Path(output_path).parent), "Output directory does not exist"

        except AssertionError as e:
            self.logger.error(f"- save - {e}")
            raise ValueError(f"Invalid parameters: {e}")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_image = self._denormalize(face_image) if denormalize else face_image

        #Convert BGR to RGB for PIL
        if len(save_image.shape) == 3 and save_image.shape[2] == 3:
            save_image_rgb = cv2.cvtColor(save_image, cv2.COLOR_BGR2RGB)
        else:
            save_image_rgb = save_image

        pil_image = Image.fromarray(save_image_rgb)
        pil_image.save(output_path, "JPEG", optimize=True)

        self.logger.info(f"- save - saved face to {output_path}")