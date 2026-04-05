from typing import List, Optional
from torch import nn
import torch
from spatial.efficient_net import EfficientNetFeatureExtractor
from frequal.frequency_feature_extractor import FrequencyFeatureExtractor
from classification.face_features_concatenator import FaceFeaturesConcatenator
from classification.face_classifier import FaceClassifier
from logging import Logger, getLogger
from utils.config_dataclasses import ModelArchitectureConfig

class DeepFakeDetector(nn.Module):
    """Deepfake detection model combining spatial and frequency analysis."""

    def __init__(self, pretrained: bool =True, use_gpu: bool =True, config: Optional[ModelArchitectureConfig] =None) -> None:
        """Initialize DeepFakeDetector.
        Parameters:
            pretrained (bool): whether to use pretrained weights for EfficientNet-B4 (legacy, overridden by config)
            use_gpu (bool): whether to use GPU for frequency feature extraction (legacy, overridden by config)
            config (ModelArchitectureConfig, optional): model architecture configuration
        Raises:
            RuntimeError: if GPU requested but CUDA is not available
        """
        super(DeepFakeDetector, self).__init__()

        self.logger: Logger = getLogger("/".join(__file__.split("/")[-2:]))
        """Logger instance for the DeepFakeDetector class"""

        if config is None:
            config = ModelArchitectureConfig()

        if pretrained is not None:
            config.spatial_features.pretrained = pretrained
        if use_gpu is not None:
            config.frequency_features.use_gpu_extractors = use_gpu

        self.config: ModelArchitectureConfig = config
        """Model architecture configuration"""

        self.efficient: EfficientNetFeatureExtractor = EfficientNetFeatureExtractor(
            pretrained=config.spatial_features.pretrained,
            path_to_weights=config.spatial_features.path_to_weights
        )
        """EfficientNet-B4 feature extractor for spatial analysis"""

        self.frequal_extractor: FrequencyFeatureExtractor = FrequencyFeatureExtractor(
            fft_feature_dim=config.frequency_features.fft.feature_dim,
            fft_num_radial_bands=config.frequency_features.fft.num_radial_bands,
            fft_window_function=config.frequency_features.fft.window_function,
            fft_high_freq_emphasis=config.frequency_features.fft.high_freq_emphasis,
            dct_feature_dim=config.frequency_features.dct.feature_dim,
            dct_block_size=config.frequency_features.dct.block_size,
            channel_mode=config.frequency_features.channel_mode,
            dct_aggregation_method=config.frequency_features.dct.aggregation_method,
            dct_num_frequency_bands=config.frequency_features.dct.num_frequency_bands,
            fusion_output_dim=config.frequency_features.fusion.output_dim,
            fusion_hidden_dims=config.frequency_features.fusion.hidden_dims,
            fusion_dropout_rate=config.frequency_features.fusion.dropout_rate,
            fusion_use_batch_norm=config.frequency_features.fusion.use_batch_norm,
            path_to_weights=config.frequency_features.fusion.path_to_weights,
            use_gpu_extractors=config.frequency_features.use_gpu_extractors,
            fft_num_azimuthal_sectors=config.frequency_features.fft_constants.num_azimuthal_sectors,
            fft_high_freq_threshold_ratio=config.frequency_features.fft_constants.high_freq_threshold_ratio,
            fft_artifact_edge_width=config.frequency_features.fft_constants.artifact_edge_width,
            fft_artifact_num_radial_samples=config.frequency_features.fft_constants.artifact_num_radial_samples,
            fft_artifact_center_region_size=config.frequency_features.fft_constants.artifact_center_region_size,
            fft_default_smoothing_kernel_size=config.frequency_features.fft_constants.default_smoothing_kernel_size,
            fft_energy_band_start_multiplier=config.frequency_features.fft_constants.energy_band_start_multiplier,
            fft_energy_band_end_multiplier=config.frequency_features.fft_constants.energy_band_end_multiplier,
            fft_epsilon=config.frequency_features.fft_constants.epsilon,
            dct_epsilon=config.frequency_features.dct_constants.epsilon
        )
        """Frequency feature extractor using FFT and DCT"""

        self.concatenator: FaceFeaturesConcatenator = FaceFeaturesConcatenator()
        """Concatenates spatial and frequency features"""

        self.classifier: FaceClassifier = FaceClassifier(
            input_dim=config.classifier.input_dim,
            num_classes=config.classifier.num_classes,
            hidden_dims=config.classifier.hidden_dims,
            dropout_rate=config.classifier.dropout_rate,
            use_batch_norm=config.classifier.use_batch_norm,
            path_to_weights=config.classifier.path_to_weights
        )
        """Face classifier taking concatenated features and outputting logits"""

    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        """Forward pass through the deepfake detection model.
        Parameters:
            x_image (torch.Tensor): Input tensor [batch, 3, 224, 224]
                - Format: RGB (not BGR)
                - Normalized with ImageNet statistics
                - Values typically in range [-2.5, 2.5] after normalization
        Returns:
            torch.Tensor: Logits [batch, 2] for [REAL, FAKE] classes
                - Raw scores (not probabilities)
                - Apply softmax for probabilities: torch.softmax(output, dim=1)
                - Apply argmax for predictions: torch.argmax(output, dim=1)
        Raises:
            AssertionError: if input shape is not (batch, 3, 224, 224)
        """
        try:
            assert x_image.shape[1:] == (3, 224, 224), f"Input must be [batch, 3, 224, 224], got {x_image.shape}"
        
        except AssertionError as e:
            err_msg = f"Invalid input shape: {e}"
            self.logger.error(err_msg)
            raise AssertionError(err_msg) from e

        # Extract spatial features from EfficientNet-B4
        # Returns: (intermediate_features, pooled_features [B, 1792])
        _, spatial_features = self.efficient(x_image)

        # Extract frequency features from FFT + DCT
        # Returns: fused_features [B, 1024]
        frequency_features = self.frequal_extractor(x_image)

        # Concatenate spatial and frequency features
        # [B, 1792] + [B, 1024] → [B, 2816]
        combined_features = self.concatenator(spatial_features, frequency_features)

        # Classification: [B, 2816] → [B, 2]
        output = self.classifier(combined_features)

        return output