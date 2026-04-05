import torch
import torch.nn as nn
import logging


class FaceFeaturesConcatenator(nn.Module):
    """This module takes pooled spatial features from EfficientNet-B4 and
    fused frequency features from FrequencyFeatureExtractor, and concatenates
    them into a single feature vector for downstream classification.
    """

    def __init__(self,
        spatial_feature_dim: int = 1792,
        frequency_feature_dim: int = 1024
    ):
        """Initialize FaceFeaturesConcatenator.
        Parameters:
            spatial_feature_dim (int): Expected spatial feature dimension from EfficientNet-B4 (default: 1792)
            frequency_feature_dim (int): Expected frequency feature dimension from FrequencyFeatureExtractor
        Raises:
            ValueError: if parameters are invalid
        """
        super(FaceFeaturesConcatenator, self).__init__()

        self.logger: logging.Logger = logging.getLogger("classification.FaceFeaturesConcatenator")
        """Logger instance for the FaceFeaturesConcatenator class"""

        try:
            assert spatial_feature_dim > 0, f"Spatial feature dimension must be positive, got {spatial_feature_dim}"
            assert frequency_feature_dim > 0, f"Frequency feature dimension must be positive, got {frequency_feature_dim}"

        except AssertionError as e:
            self.logger.fatal(f"Invalid parameters: {e}")
            raise ValueError(f"Invalid parameters: {e}")

        self.spatial_feature_dim: int = spatial_feature_dim
        """Expected spatial feature dimension (1792 for EfficientNet-B4)"""
        self.frequency_feature_dim: int = frequency_feature_dim
        """Expected frequency feature dimension"""
        self.output_dim: int = spatial_feature_dim + frequency_feature_dim
        """Total concatenated feature dimension"""


        # Use LayerNorm instead of BatchNorm to support batch_size=1
        self.normalization: nn.LayerNorm = nn.LayerNorm(self.output_dim)
        """Layer normalization to normalize concatenated features"""

        # Initialize normalization weights
        self._initialize_weights()

        self.logger.info(f"FaceFeaturesConcatenator initialized: spatial_dim={spatial_feature_dim}, "
                        f"frequency_dim={frequency_feature_dim}, output_dim={self.output_dim}")

    def _initialize_weights(self) -> None:
        """Initialize normalization layer weights."""
        nn.init.constant_(self.normalization.weight, 1)
        nn.init.constant_(self.normalization.bias, 0)
        self.logger.info("LayerNorm weights initialized")

    def forward(self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate spatial and frequency features.
        Parameters:
            spatial_features (torch.Tensor): Pooled spatial features from EfficientNet
                - Shape: [batch, spatial_feature_dim, 1, 1] or [batch, spatial_feature_dim]
            frequency_features (torch.Tensor): Fused frequency features from FrequencyFeatureExtractor
                - Shape: [batch, frequency_feature_dim]
        Returns:
            torch.Tensor: Concatenated features
                - Shape: [batch, output_dim] where output_dim = spatial_feature_dim + frequency_feature_dim
        Raises:
            ValueError: if input shapes are invalid
        """
        try:
            #flatten spatial features if needed (from [batch, dim, 1, 1] to [batch, dim])
            if len(spatial_features.shape) == 4:
                assert spatial_features.shape[2] == 1 and spatial_features.shape[3] == 1, \
                    f"Spatial features must be pooled to shape [batch, dim, 1, 1], got {spatial_features.shape}"
                spatial_features = spatial_features.squeeze(-1).squeeze(-1)

            #validate shapes
            assert len(spatial_features.shape) == 2, \
                f"Spatial features must be 2D [batch, dim], got shape {spatial_features.shape}"
            assert len(frequency_features.shape) == 2, \
                f"Frequency features must be 2D [batch, dim], got shape {frequency_features.shape}"

            batch_size_spatial = spatial_features.shape[0]
            batch_size_freq = frequency_features.shape[0]
            assert batch_size_spatial == batch_size_freq, \
                f"Batch sizes must match: spatial={batch_size_spatial}, frequency={batch_size_freq}"

            assert spatial_features.shape[1] == self.spatial_feature_dim, \
                f"Expected spatial feature dim {self.spatial_feature_dim}, got {spatial_features.shape[1]}"
            assert frequency_features.shape[1] == self.frequency_feature_dim, \
                f"Expected frequency feature dim {self.frequency_feature_dim}, got {frequency_features.shape[1]}"

        except AssertionError as e:
            self.logger.error(f"- forward - {e}")
            raise ValueError(f"Invalid input shape: {e}")

        #concatenate features along feature dimension
        concatenated = torch.cat([spatial_features, frequency_features], dim=1)

        # Apply layer normalization to equalize feature scales
        concatenated = self.normalization(concatenated)

        self.logger.debug(f"- forward - Concatenated features: spatial {spatial_features.shape} + "
                         f"frequency {frequency_features.shape} = {concatenated.shape}")

        return concatenated
