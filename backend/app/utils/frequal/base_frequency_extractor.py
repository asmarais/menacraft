import logging
import torch.nn as nn


class BaseFrequencyExtractor:
    """Base class for all frequency-based feature extractors.
    Provides common configuration for FFT, DCT, and other frequency domain extractors.
    Subclasses should implement their own extract() method appropriate for their data type."""

    AVAILABLE_CHANNEL_MODES = ["luminance", "grayscale", "per_channel"]
    """Available channel processing modes"""

    def __init__(self,
        channel_mode: str,
        feature_dim: int,
        logger_name: str
    ):
        """Initialize BaseFrequencyExtractor.
        Parameters:
            channel_mode (str): how to handle color channels
            feature_dim (int): target output feature dimension
            logger_name (str): name for the logger instance
        Raises:
            ValueError: if parameters are invalid
        """
        self.logger: logging.Logger = logging.getLogger(logger_name)
        """Logger instance for the extractor"""

        try:
            assert channel_mode in self.AVAILABLE_CHANNEL_MODES, f"Invalid channel mode: {channel_mode}"
            assert feature_dim > 0, "Feature dimension must be positive"

        except AssertionError as e:
            self.logger.fatal(f"Invalid parameters: {e}")
            raise ValueError(f"Invalid parameters: {e}")

        self.channel_mode: str = channel_mode
        """Channel processing mode"""
        self.feature_dim: int = feature_dim
        """Target output feature dimension"""
        self._num_raw_features: int = None
        """Number of raw features (set by derived classes)"""
        self.projection: nn.Module = None
        """Learnable projection from raw features to target dimension (set by derived classes)"""
