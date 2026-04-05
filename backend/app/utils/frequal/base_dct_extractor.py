import logging
import torch
from .base_frequency_extractor import BaseFrequencyExtractor


class BaseDCTExtractor(BaseFrequencyExtractor):
    """Base class for DCT-based frequency feature extractors.
    Provides common configuration for both NumPy and PyTorch implementations
    of DCT feature extraction for deepfake detection."""

    AVAILABLE_AGGREGATION_METHODS = ["zigzag", "frequency_bands", "statistical"]
    """Available methods for aggregating DCT block features (implementation constant)"""
    DEFAULT_BLOCK_SIZE = 8
    """Default DCT block size (JPEG standard, implementation constant)"""

    def __init__(self,
        block_size: int,
        channel_mode: str,
        aggregation_method: str,
        num_frequency_bands: int,
        feature_dim: int,
        logger_name: str,
        epsilon: float = 1e-10
    ):
        """Initialize BaseDCTExtractor.
        Parameters:
            block_size (int): size of DCT blocks in pixels
            channel_mode (str): how to handle color channels
            aggregation_method (str): how to aggregate block statistics
            num_frequency_bands (int): number of frequency bands to extract
            feature_dim (int): target output feature dimension
            logger_name (str): name for the logger instance
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
            assert block_size in [8, 16], f"Block size must be 8 or 16, got {block_size}"
            assert aggregation_method in self.AVAILABLE_AGGREGATION_METHODS, f"Invalid aggregation method: {aggregation_method}"
            assert num_frequency_bands > 0, "Number of frequency bands must be positive"
            assert epsilon > 0, "Epsilon must be positive"

        except AssertionError as e:
            self.logger.fatal(f"Invalid parameters: {e}")
            raise ValueError(f"Invalid parameters: {e}")

        self.block_size: int = block_size
        """Size of DCT blocks in pixels"""
        self.aggregation_method: str = aggregation_method
        """Method for aggregating block features"""
        self.num_frequency_bands: int = num_frequency_bands
        """Number of frequency bands for aggregation"""
        self.epsilon: float = epsilon
        """Small constant for numerical stability"""
        self.EPSILON: float = epsilon
        """Small constant for numerical stability (uppercase alias)"""

        self._num_raw_features: int = self._calculate_raw_feature_dim()

        self.projection: torch.nn.Linear = torch.nn.Linear(self._num_raw_features, feature_dim)
        """Learnable projection from raw features to target dimension"""

    def _calculate_raw_feature_dim(self) -> int:
        """Calculate the number of raw features before projection.
        Returns:
            int: number of raw features
        Raises:
            ValueError: if aggregation method is unknown
        """
        match self.aggregation_method:
            case "frequency_bands":
                #bands × 3 statistics + 5 global stats + 10 histogram features
                return self.num_frequency_bands * 3 + 5 + 10
            case "zigzag":
                #top 16 coefficients + 10 histogram bins
                return 16 + 10
            case "statistical":
                #block_size^2 positions × 3 statistics
                return (self.block_size ** 2) * 3
            case _:
                self.logger.fatal(f"Unknown aggregation method: {self.aggregation_method}")
                raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
