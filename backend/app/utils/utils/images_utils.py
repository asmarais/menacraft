"""
Image manipulation utilities for VeridisQuo.

This module provides generic image processing functions including:
- Color space conversions (BGR, grayscale, luminance)
- Format checking and validation
- Tensor to array conversions
"""

import cv2
import numpy as np
import torch
from logging import Logger, getLogger

logger: Logger = getLogger("/".join(__file__.split("/")[-2:]))  

def convert_to_luminance(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR image to luminance (Y channel of YCbCr).
    Parameters:
        image (np.ndarray): input image [height, width, 3] in BGR format
    Returns:
        np.ndarray: luminance channel [height, width]
    Raises:
        ValueError: if image format is invalid
    """
    try:
        if is_grayscale(image):
            return image
        assert is_bgr(image), "Image must have 3 channels (BGR)"

    except AssertionError as e:
        logger.error(f"Invalid image for luminance conversion: {e}")
        raise ValueError(f"Invalid image: {e}")

    #convert BGR to YCrCb and extract Y channel (luminance)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0]


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR image to grayscale.
    Parameters:
        image (np.ndarray): input image [height, width, 3] in BGR format
    Returns:
        np.ndarray: grayscale image [height, width]
    Raises:
        ValueError: if image format is invalid
    """
    try:
        if is_grayscale(image):
            return image
        assert is_bgr(image), "Image must have 3 channels (BGR)"

    except AssertionError as e:
        logger.error(f"Invalid image for grayscale conversion: {e}")
        raise ValueError(f"Invalid image: {e}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_per_channel(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR image to per-channel average representation.
    Parameters:
        image (np.ndarray): input image [height, width, 3] in BGR format
    Returns:
        np.ndarray: averaged channel image [height, width]
    Raises:
        ValueError: if image format is invalid
    """
    try:
        if is_grayscale(image):
            return image
        assert is_bgr(image), "Image must have 3 channels (BGR)"

    except AssertionError as e:
        logger.error(f"Invalid image for per-channel conversion: {e}")
        raise ValueError(f"Invalid image: {e}")

    #Average across all channels (preserves more info than weighted conversion)
    #This gives equal weight to all color channels for frequency analysis
    channel_avg = np.mean(image, axis=2).astype(image.dtype)
    return channel_avg


def is_grayscale(image: np.ndarray) -> bool:
    """Check if the input image is grayscale.
    Parameters:
        image (np.ndarray): input image
    Returns:
        bool: True if the image is grayscale, False otherwise (including non valid images)
    """
    if image is None or image.size == 0:
        return False
    return len(image.shape) == 2 or image.shape[2] == 1

def is_bgr(image: np.ndarray) -> bool:
    """Check if the input image is in BGR format.
    Parameters:
        image (np.ndarray): input image
    Returns:
        bool: True if the image is in BGR format, False otherwise (including non valid images)
    """
    if image is None or image.size == 0:
        return False
    return len(image.shape) == 3 and image.shape[2] == 3

def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array.
    PyTorch's [batch, C, H, W] RGB format to OpenCV's [batch, H, W, C] BGR format
    in [0, 255] range.
    Parameters:
        tensor (torch.Tensor): input tensor [batch, channels, height, width]
            - Values in [0, 1] or [-1, 1]
    Returns:
        np.ndarray: NumPy array [batch, height, width, channels] in BGR format
            - Values in [0, 255] as uint8
    Raises:
        ValueError: if tensor format is invalid
    """
    try:
        assert tensor is not None, "Tensor cannot be None"
        assert len(tensor.shape) == 4, f"Tensor must be 4D [batch, C, H, W], got shape {tensor.shape}"
        assert tensor.shape[1] == 3, f"Tensor must have 3 channels, got {tensor.shape[1]}"

    except AssertionError as e:
        logger.error(f"Invalid tensor: {e}")
        raise ValueError(f"Invalid tensor: {e}")

    # Detach from computation graph and move to CPU
    numpy_array = tensor.detach().cpu().numpy()

    # Transpose [batch, C, H, W] to [batch, H, W, C]
    numpy_array = np.transpose(numpy_array, (0, 2, 3, 1))

    # Process each image in batch
    batch_size = numpy_array.shape[0]
    result = []
    for i in range(batch_size):
        img = numpy_array[i]

        # Format image to [0, 255] range
        if img.dtype == np.float32 or img.dtype == np.float64:
            # Denormalize to [0, 255] based on value range
            if img.max() <= 1.0 and img.min() >= 0.0:
                # Normalized to [0, 1]
                img = (img * 255.0).astype(np.uint8)
            elif img.min() < 0:
                # Normalized to [-1, 1]
                img = ((img + 1.0) * 127.5).astype(np.uint8)
            else:
                # Already in [0, 255] range
                img = img.astype(np.uint8)

        # OpenCV handles BGR, not RGB - reverse channel order
        img_bgr = img[:, :, ::-1].copy()
        result.append(img_bgr)

    numpy_array_bgr = np.stack(result, axis=0)

    logger.debug(f"Converted tensor to NumPy array with shape {numpy_array_bgr.shape}")
    return numpy_array_bgr
