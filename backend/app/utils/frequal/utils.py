"""
Frequency analysis utilities for VeridisQuo.

This module provides utilities for frequency-domain analysis including:
- Color space conversions (delegated to utils.images_utils)
- Window functions for FFT preprocessing
- Spectral analysis utilities
- JPEG-style zigzag ordering for DCT
"""

import cv2
import numpy as np
from typing import Tuple, Dict
import logging
import torch

# Import generic image utilities from utils module
from utils.images_utils import (
    convert_to_luminance as _convert_to_luminance,
    convert_to_grayscale as _convert_to_grayscale,
    convert_per_channel as _convert_per_channel,
    is_grayscale as _is_grayscale,
    is_bgr as _is_bgr,
    torch_to_numpy as _torch_to_numpy
)

logger: logging.Logger = logging.getLogger("/".join(__file__.split("/")[-2:]))

def convert_to_luminance(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR image to luminance (Y channel of YCbCr).
    This is a wrapper around utils.images_utils.convert_to_luminance.
    Parameters:
        image (np.ndarray): input image [height, width, 3] in BGR format
    Returns:
        np.ndarray: luminance channel [height, width]
    Raises:
        ValueError: if image format is invalid
    """
    return _convert_to_luminance(image)

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR image to grayscale.
    This is a wrapper around utils.images_utils.convert_to_grayscale.
    Parameters:
        image (np.ndarray): input image [height, width, 3] in BGR format
    Returns:
        np.ndarray: grayscale image [height, width]
    Raises:
        ValueError: if image format is invalid
    """
    return _convert_to_grayscale(image)

def convert_per_channel(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR image to per-channel average representation.
    This is a wrapper around utils.images_utils.convert_per_channel.
    Parameters:
        image (np.ndarray): input image [height, width, 3] in BGR format
    Returns:
        np.ndarray: averaged channel image [height, width]
    Raises:
        ValueError: if image format is invalid
    """
    return _convert_per_channel(image)

def is_grayscale(image: np.ndarray) -> bool:
    """Check if the input image is grayscale.
    This is a wrapper around utils.images_utils.is_grayscale.
    Parameters:
        image (np.ndarray): input image
    Returns:
        bool: True if the image is grayscale, False otherwise (including non valid images)
    """
    return _is_grayscale(image)

def is_bgr(image: np.ndarray) -> bool:
    """Check if the input image is in BGR format.
    This is a wrapper around utils.images_utils.is_bgr.
    Parameters:
        image (np.ndarray): input image
    Returns:
        bool: True if the image is in BGR format, False otherwise (including non valid images)
    """
    return _is_bgr(image)

def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array.
    This is a wrapper around utils.images_utils.torch_to_numpy.
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
    return _torch_to_numpy(tensor)

def create_2d_window(shape: Tuple[int, int], window_type: str = "hann") -> np.ndarray:
    """Create 2D window function for FFT preprocessing to reduce spectral leakage.
    Parameters:
        shape (Tuple[int, int]): window shape (height, width)
        window_type (str): type of window function
            - "hann": Hann window (recommended for general use)
            - "hamming": Hamming window
            - "blackman": Blackman window
            - None: no windowing (rectangular window)
    Returns:
        np.ndarray: 2D window function [height, width]
    Raises:
        ValueError: if parameters are invalid
    """
    try:
        assert len(shape) == 2, "Shape must be (height, width)"
        assert shape[0] > 0 and shape[1] > 0, "Shape dimensions must be positive"

    except AssertionError as e:
        logger.error(f"Invalid parameters for window creation: {e}")
        raise ValueError(f"Invalid parameters: {e}")

    h, w = shape

    match window_type:
        case "hann":
            window_h = np.hanning(h)
            window_w = np.hanning(w)
        case "hamming":
            window_h = np.hamming(h)
            window_w = np.hamming(w)
        case "blackman":
            window_h = np.blackman(h)
            window_w = np.blackman(w)
        case None:
            return np.ones((h, w), dtype=np.float32)
        case _:
            logger.error(f"Unknown window type: {window_type}")
            raise ValueError(f"Unknown window type: {window_type}")

    window_2d = np.outer(window_h, window_w)
    return window_2d.astype(np.float32)

def create_radial_mask(center: Tuple[int, int], shape: Tuple[int, int], radius: float) -> np.ndarray:
    """Create circular mask for radial frequency band extraction.
    Parameters:
        center (Tuple[int, int]): center coordinates (cy, cx)
        shape (Tuple[int, int]): mask shape (height, width)
        radius (float): radius of the circle in pixels
    Returns:
        np.ndarray: binary mask [height, width] with True inside circle
    Raises:
        ValueError: if parameters are invalid
    """
    try:
        assert len(center) == 2, "Center must be (cy, cx)"
        assert len(shape) == 2, "Shape must be (height, width)"
        assert radius >= 0, "Radius must be non-negative"

    except AssertionError as e:
        logger.error(f"Invalid parameters for radial mask: {e}")
        raise ValueError(f"Invalid parameters: {e}")

    cy, cx = center
    h, w = shape

    # Coordinate grids
    y, x = np.ogrid[:h, :w]
    # Distance from center
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)

    return distance <= radius

def zigzag_indices(n: int) -> np.ndarray:
    """Generate zigzag scan indices for n*n block (JPEG-style DCT coefficient ordering).
    From the top-left (DC coefficient) and alternating between diagonal directions to
    traverse the block in order of increasing spatial frequency.
    Parameters:
        n (int): block size (n x n)
    Returns:
        np.ndarray: zigzag indices array of shape [n*n, 2] with (row, col) pairs
    Raises:
        ValueError: if parameters are invalid
    """
    try:
        assert n > 0, "Block size must be positive"

    except AssertionError as e:
        logger.error(f"Invalid parameters for zigzag indices: {e}")
        raise ValueError(f"Invalid parameters: {e}")

    indices = []

    # Traverse diagonals
    for sum_val in range(2 * n - 1):
        if sum_val % 2 == 0:
            # Even diagonals: traverse bottom-left to top-right
            for i in range(min(sum_val, n - 1), max(-1, sum_val - n), -1):
                j = sum_val - i
                if 0 <= i < n and 0 <= j < n:
                    indices.append((i, j))
        else:
            # Odd diagonals: traverse top-right to bottom-left
            for i in range(max(0, sum_val - n + 1), min(sum_val + 1, n)):
                j = sum_val - i
                if 0 <= i < n and 0 <= j < n:
                    indices.append((i, j))

    return np.array(indices, dtype=np.int32)

def compute_spectral_stats(spectrum: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive spectral statistics from magnitude spectrum.
    These statistics characterize the frequency distribution and are useful
    for detecting anomalies in deepfake images.
    Parameters:
        spectrum (np.ndarray): magnitude spectrum (any shape)
    Returns:
        Dict[str, float]: dictionary with spectral statistics
            - "centroid": spectral centroid (center of mass in frequency domain)
            - "rolloff": spectral rolloff (95% energy threshold)
            - "flatness": spectral flatness (measure of tonality vs noise)
            - "entropy": spectral entropy (measure of randomness)
            - "peak_freq": peak frequency location (index of maximum magnitude)
    Raises:
        ValueError: if spectrum is invalid
    """
    try:
        assert spectrum is not None and spectrum.size > 0, "Empty or None spectrum"

    except AssertionError as e:
        logger.error(f"Invalid spectrum for statistics computation: {e}")
        raise ValueError(f"Invalid spectrum: {e}")

    # Flatten spectrum for 1D analysis
    flat_spectrum = spectrum.flatten()

    # Ensure non-negative values
    flat_spectrum = np.abs(flat_spectrum)

    # Total energy
    total_energy = np.sum(flat_spectrum)

    # Handle zero-energy case
    if total_energy == 0 or np.isnan(total_energy):
        return {
            "centroid": 0.0,
            "rolloff": 0.0,
            "flatness": 0.0,
            "entropy": 0.0,
            "peak_freq": 0
        }

    # Normalize to probability distribution
    prob_dist = flat_spectrum / total_energy

    # Spectral centroid (center of mass)
    indices = np.arange(len(flat_spectrum))
    centroid = np.sum(indices * prob_dist)

    # Spectral rolloff (95% of cumulative energy)
    cumsum = np.cumsum(prob_dist)
    rolloff_idx = np.where(cumsum >= 0.95)[0]
    rolloff = float(rolloff_idx[0]) if len(rolloff_idx) > 0 else float(len(flat_spectrum) - 1)

    # Spectral flatness (geometric mean / arithmetic mean)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    geometric_mean = np.exp(np.mean(np.log(flat_spectrum + epsilon)))
    arithmetic_mean = np.mean(flat_spectrum)
    flatness = geometric_mean / (arithmetic_mean + epsilon)

    # Spectral entropy
    # Filter out zero probabilities for entropy calculation
    prob_nonzero = prob_dist[prob_dist > epsilon]
    entropy = -np.sum(prob_nonzero * np.log2(prob_nonzero + epsilon))

    # Peak frequency
    peak_freq = int(np.argmax(flat_spectrum))

    return {
        "centroid": float(centroid),
        "rolloff": float(rolloff),
        "flatness": float(flatness),
        "entropy": float(entropy),
        "peak_freq": peak_freq
    }

def format_image(image: np.ndarray) -> np.ndarray:
    """Format input image to ensure it is in the correct shape and type.
    OpenCV needs images with BGR or grayscale format [H, W, C] in [0, 255] range.
    Parameters:
        image (np.ndarray): input image
    Returns:
        np.ndarray: formatted BGR image [H, W, C] in [0, 255] range
    Raises:
        ValueError: if image format is invalid
    """
    try:
        assert is_bgr(image), "Image must have 3 channels (BGR)"

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.dtype == np.float32 or image.dtype == np.float64:
            # Denormalize to [0, 255] based on value range
            if image.max() <= 1.0 and image.min() >= 0.0:
                # Normalized to [0, 1]
                image = (image * 255.0).astype(np.uint8)

            elif image.min() < 0:
                # Normalized to [-1, 1]
                image = ((image + 1.0) * 127.5).astype(np.uint8)

            else:
                # Already in [0, 255] range
                image = image.astype(np.uint8)

    except AssertionError as e:
        logger.error(f"Invalid image for formatting: {e}")
        raise ValueError(f"Invalid image: {e}")

    logger.debug(f"Formatted image successfully with shape {image.shape} and dtype {image.dtype}")
    return image
