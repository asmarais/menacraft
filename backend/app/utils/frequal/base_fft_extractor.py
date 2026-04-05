import logging
from typing import Tuple
import torch.nn as nn
import math
from .base_frequency_extractor import BaseFrequencyExtractor


class BaseFFTExtractor(BaseFrequencyExtractor):
    """Base class for FFT-based frequency feature extractors.
    Provides common configuration and utilities for both NumPy and PyTorch implementations
    of FFT feature extraction for deepfake detection."""

    AVAILABLE_WINDOW_FUNCTIONS = ["hann", "hamming", "blackman", "none", None]
    """Available window functions to reduce spectral leakage (implementation constant)"""

    def __init__(self,
        channel_mode: str,
        num_radial_bands: int,
        window_function: str,
        high_freq_emphasis: bool,
        feature_dim: int,
        logger_name: str,
        num_azimuthal_sectors: int = 8,
        high_freq_threshold_ratio: float = 0.5,
        artifact_edge_width: int = 5,
        artifact_num_radial_samples: int = 50,
        artifact_center_region_size: int = 20,
        default_smoothing_kernel_size: int = 5,
        energy_band_start_multiplier: int = 2,
        energy_band_end_multiplier: int = 3,
        epsilon: float = 1e-10
    ):
        """Initialize BaseFFTExtractor.
        Parameters:
            channel_mode (str): how to handle color channels
            num_radial_bands (int): number of radial frequency bands
            window_function (str): windowing function to reduce spectral leakage
            high_freq_emphasis (bool): emphasize high frequencies where deepfakes often have anomalies
            feature_dim (int): target output feature dimension
            logger_name (str): name for the logger instance
            num_azimuthal_sectors (int): number of angular sectors for directional analysis
            high_freq_threshold_ratio (float): ratio defining high-frequency region
            artifact_edge_width (int): width of edge regions for artifact detection
            artifact_num_radial_samples (int): number of radial samples for upsampling detection
            artifact_center_region_size (int): half-size of center region for variance analysis
            default_smoothing_kernel_size (int): default kernel size for spectrum smoothing
            energy_band_start_multiplier (int): multiplier for energy band start index
            energy_band_end_multiplier (int): multiplier for energy band end index
            epsilon (float): small constant for numerical stability
        Raises:
            ValueError: if parameters are invalid
        """
        super().__init__(
            channel_mode=channel_mode,
            feature_dim=feature_dim,
            logger_name=logger_name
        )

        try:
            assert num_radial_bands > 0, "Number of radial bands must be positive"
            assert window_function in self.AVAILABLE_WINDOW_FUNCTIONS, f"Invalid window function: {window_function}"
            assert num_azimuthal_sectors > 0, "Number of azimuthal sectors must be positive"
            assert 0 < high_freq_threshold_ratio < 1, "High frequency threshold ratio must be in (0, 1)"
            assert epsilon > 0, "Epsilon must be positive"

        except AssertionError as e:
            self.logger.fatal(f"Invalid parameters: {e}")
            raise ValueError(f"Invalid parameters: {e}")

        self.num_radial_bands: int = num_radial_bands
        """Number of radial frequency bands"""
        self.window_function: str = window_function
        """Window function type"""
        self.high_freq_emphasis: bool = high_freq_emphasis
        """Whether to emphasize high frequencies"""

        self.num_azimuthal_sectors: int = num_azimuthal_sectors
        """Number of angular sectors for directional analysis"""
        self.NUM_AZIMUTHAL_SECTORS: int = num_azimuthal_sectors
        """Number of angular sectors for directional analysis (uppercase alias)"""

        self.high_freq_threshold_ratio: float = high_freq_threshold_ratio
        """Ratio defining high-frequency region"""
        self.HIGH_FREQ_THRESHOLD_RATIO: float = high_freq_threshold_ratio
        """Ratio defining high-frequency region (uppercase alias)"""

        self.artifact_edge_width: int = artifact_edge_width
        """Width of edge regions for artifact detection"""
        self.ARTIFACT_EDGE_WIDTH: int = artifact_edge_width
        """Width of edge regions for artifact detection (uppercase alias)"""

        self.artifact_num_radial_samples: int = artifact_num_radial_samples
        """Number of radial samples for upsampling detection"""
        self.ARTIFACT_NUM_RADIAL_SAMPLES: int = artifact_num_radial_samples
        """Number of radial samples for upsampling detection (uppercase alias)"""

        self.artifact_center_region_size: int = artifact_center_region_size
        """Half-size of center region for variance analysis"""
        self.ARTIFACT_CENTER_REGION_SIZE: int = artifact_center_region_size
        """Half-size of center region for variance analysis (uppercase alias)"""

        self.default_smoothing_kernel_size: int = default_smoothing_kernel_size
        """Default kernel size for spectrum smoothing"""
        self.DEFAULT_SMOOTHING_KERNEL_SIZE: int = default_smoothing_kernel_size
        """Default kernel size for spectrum smoothing (uppercase alias)"""

        self.energy_band_start_multiplier: int = energy_band_start_multiplier
        """Multiplier for energy band start index"""
        self.ENERGY_BAND_START_MULTIPLIER: int = energy_band_start_multiplier
        """Multiplier for energy band start index (uppercase alias)"""

        self.energy_band_end_multiplier: int = energy_band_end_multiplier
        """Multiplier for energy band end index"""
        self.ENERGY_BAND_END_MULTIPLIER: int = energy_band_end_multiplier
        """Multiplier for energy band end index (uppercase alias)"""

        self.epsilon: float = epsilon
        """Small constant for numerical stability"""
        self.EPSILON: float = epsilon
        """Small constant for numerical stability (uppercase alias)"""

        self._num_raw_features: int = self._calculate_raw_feature_dim()

        self.projection: nn.Linear = nn.Linear(self._num_raw_features, feature_dim)
        """Learnable projection from raw features to target dimension"""

    def _calculate_raw_feature_dim(self) -> int:
        """Calculate the number of raw features before projection, specifically for deepfake detection.
        Returns:
            int: number of raw features
        """
        # Radial bands: 8 bands × 4 statistics = 32
        # Radial profile: 8 values = 8
        # Azimuthal: 8 directions × 2 statistics = 16
        # Global spectral stats: 8
        # High-freq artifacts: 6
        # Cross-band interactions: 5
        return (self.num_radial_bands * 4) + self.num_radial_bands + 16 + 8 + 6 + 5

    @staticmethod
    def _compute_max_radius(center: Tuple[int, int]) -> float:
        """Compute maximum radius from center to corner.
        Parameters:
            center (Tuple[int, int]): center coordinates (cy, cx)
        Returns:
            float: maximum radius (distance to corner)
        """
        return math.sqrt(center[0]**2 + center[1]**2)

    def _get_sector_width(self) -> float:
        """Get angular width of each azimuthal sector.
        Returns:
            float: sector width in radians
        """
        return 2 * math.pi / self.num_azimuthal_sectors
