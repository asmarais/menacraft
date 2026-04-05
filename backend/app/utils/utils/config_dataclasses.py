"""Configuration dataclasses for VeridisQuo deepfake detection system.

This module provides a comprehensive configuration hierarchy for all components:
- Model architecture (frequency features, spatial features, classifier)
- Preprocessing (face detection, face extraction, frame extraction)
- Inference options
- Training configuration

All parameters including algorithmic constants are exposed for maximum flexibility.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Union, Dict, Any
from pathlib import Path
import torch


@dataclass
class FFTConfig:
    """Configuration for FFT feature extraction."""

    feature_dim: int = 512
    """Output feature dimension after projection."""

    num_radial_bands: int = 8
    """Number of radial frequency bands (JPEG-style sectors)."""

    window_function: str = "hann"
    """Window function for spectral leakage reduction: 'hann', 'hamming', 'blackman', 'none', or None."""

    high_freq_emphasis: bool = True
    """Whether to emphasize high frequency components."""


@dataclass
class FFTConstantsConfig:
    """Algorithmic constants for FFT feature extraction."""

    num_azimuthal_sectors: int = 8
    """Number of azimuthal sectors (45-degree sectors)."""

    high_freq_threshold_ratio: float = 0.5
    """Threshold ratio for high frequency region (outer 50% of spectrum)."""

    artifact_edge_width: int = 5
    """Edge width for artifact detection."""

    artifact_num_radial_samples: int = 50
    """Number of radial samples for artifact detection."""

    artifact_center_region_size: int = 20
    """Center region size for artifact detection."""

    default_smoothing_kernel_size: int = 5
    """Default smoothing kernel size."""

    energy_band_start_multiplier: int = 2
    """Start multiplier for energy band calculation."""

    energy_band_end_multiplier: int = 3
    """End multiplier for energy band calculation."""

    epsilon: float = 1e-10
    """Small constant to avoid division by zero."""


@dataclass
class DCTConfig:
    """Configuration for DCT feature extraction."""

    feature_dim: int = 512
    """Output feature dimension after projection."""

    block_size: int = 8
    """DCT block size in pixels (8 or 16 only, JPEG standard is 8)."""

    aggregation_method: str = "frequency_bands"
    """Aggregation method: 'zigzag', 'frequency_bands', or 'statistical'."""

    num_frequency_bands: int = 4
    """Number of frequency bands for 'frequency_bands' aggregation method."""


@dataclass
class DCTConstantsConfig:
    """Algorithmic constants for DCT feature extraction."""

    epsilon: float = 1e-10
    """Small constant to avoid division by zero."""

    default_block_size: int = 8
    """Default DCT block size (JPEG standard)."""


@dataclass
class FusionMLPConfig:
    """Configuration for MLP fusion of frequency features."""

    output_dim: Optional[int] = None
    """Output dimension. If None, uses combined_dim (dct_feature_dim + fft_feature_dim)."""

    hidden_dims: Optional[List[int]] = None
    """Hidden layer dimensions. If None, uses [combined_dim, combined_dim // 2]."""

    dropout_rate: float = 0.3
    """Dropout probability for regularization."""

    use_batch_norm: bool = True
    """Whether to use batch normalization (actually LayerNorm for batch_size=1 support)."""

    path_to_weights: Optional[str] = None
    """Path to pretrained weights for fusion MLP."""


@dataclass
class FrequencyFeaturesConfig:
    """Configuration for frequency feature extraction (FFT + DCT)."""

    fft: FFTConfig = field(default_factory=FFTConfig)
    """FFT extractor configuration."""

    fft_constants: FFTConstantsConfig = field(default_factory=FFTConstantsConfig)
    """FFT algorithmic constants."""

    dct: DCTConfig = field(default_factory=DCTConfig)
    """DCT extractor configuration."""

    dct_constants: DCTConstantsConfig = field(default_factory=DCTConstantsConfig)
    """DCT algorithmic constants."""

    fusion: FusionMLPConfig = field(default_factory=FusionMLPConfig)
    """Fusion MLP configuration."""

    channel_mode: str = "luminance"
    """Channel mode: 'luminance', 'grayscale', or 'per_channel'."""

    use_gpu_extractors: bool = True
    """Whether to use PyTorch (GPU) extractors instead of NumPy (CPU)."""


@dataclass
class SpatialFeaturesConfig:
    """Configuration for spatial feature extraction (EfficientNet)."""

    pretrained: bool = True
    """Whether to use ImageNet pretrained weights for EfficientNet-B4."""

    path_to_weights: Optional[str] = None
    """Path to custom pretrained weights. If None, uses torchvision defaults."""


@dataclass
class ClassifierConfig:
    """Configuration for face classifier."""

    input_dim: int = 2816
    """Input feature dimension (1792 spatial + 1024 frequency by default)."""

    num_classes: int = 2
    """Number of output classes (2 for REAL/FAKE binary classification)."""

    hidden_dims: Optional[List[int]] = None
    """Hidden layer dimensions. If None, uses [1024, 512, 256]."""

    dropout_rate: float = 0.2
    """Dropout probability for regularization."""

    use_batch_norm: bool = True
    """Whether to use batch normalization (actually LayerNorm for batch_size=1 support)."""

    path_to_weights: Optional[str] = None
    """Path to pretrained classifier weights."""


@dataclass
class ModelArchitectureConfig:
    """Configuration for complete model architecture."""

    frequency_features: FrequencyFeaturesConfig = field(default_factory=FrequencyFeaturesConfig)
    """Frequency feature extraction configuration (FFT + DCT + Fusion)."""

    spatial_features: SpatialFeaturesConfig = field(default_factory=SpatialFeaturesConfig)
    """Spatial feature extraction configuration (EfficientNet-B4)."""

    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    """Classifier configuration."""


@dataclass
class FaceDetectionConfig:
    """Configuration for face detection."""

    enabled: bool = True
    """Whether face detection is enabled."""

    model_path: Optional[str] = None
    """Path to YOLO face detection model. If None, uses default from paths config."""

    min_face_size: int = 40
    """Minimum face size in pixels to accept detection."""

    confidence_threshold: float = 0.7
    """Minimum confidence score to accept detection."""

    only_keep_top: bool = True
    """If True and multiple faces detected, keep only highest confidence face."""


@dataclass
class FaceExtractionConfig:
    """Configuration for face extraction and preprocessing."""

    target_size: Tuple[int, int] = (224, 224)
    """Target output size for face images (width, height)."""

    normalization: Optional[str] = None
    """Normalization method: None, 'zero_one', or 'minus_one_one'.
    None means no normalization here (handled by ImageNet normalization later)."""

    padding: int = 0
    """Padding in pixels around detected face bounding box."""


@dataclass
class FrameExtractionConfig:
    """Configuration for video frame extraction."""

    frames_per_second: Optional[int] = 1
    """Number of frames to sample per second. If None, extracts all frames."""

    use_optimized_extractor: bool = True
    """Whether to use OptimizedFramesExtractor (PyAV with hardware acceleration)."""

    num_save_threads: int = 4
    """Number of threads for parallel frame saving (OptimizedFramesExtractor only)."""

    batch_size: int = 32
    """Batch size for frame processing (OptimizedFramesExtractor only)."""


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""

    face_detection: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    """Face detection configuration."""

    face_extraction: FaceExtractionConfig = field(default_factory=FaceExtractionConfig)
    """Face extraction configuration."""

    frame_extraction: FrameExtractionConfig = field(default_factory=FrameExtractionConfig)
    """Frame extraction configuration for videos."""

    imagenet_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    """ImageNet mean for normalization (applied after face extraction)."""

    imagenet_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    """ImageNet std for normalization (applied after face extraction)."""


@dataclass
class DeviceConfig:
    """Configuration for device selection."""

    type: Optional[str] = None
    """Device type: None (auto), 'cpu', 'cuda', 'cuda:0', etc."""

    def get_device(self) -> torch.device:
        """Get torch device based on config.
        Returns:
            torch.device: configured device or auto-selected device
        """
        if self.type is None or self.type == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.type)


@dataclass
class PathsConfig:
    """Configuration for file paths."""

    model_path: Optional[str] = None
    """Path to deepfake detection model weights (.pth file)."""

    face_detection_model_path: Optional[str] = None
    """Path to face detection model (YOLO). If None, uses default from config.py."""

    project_root: Optional[str] = None
    """Project root directory. If None, uses config.PROJECT_ROOT."""


@dataclass
class InferenceOptionsConfig:
    """Configuration for inference behavior."""

    auto_download_model: bool = True
    """Whether to automatically download model from HuggingFace if not found."""

    video_aggregation_method: str = "majority"
    """Method to aggregate frame predictions: 'majority', 'average', 'weighted_average',
    'max_confidence', or 'threshold'."""

    inference_batch_size: int = 8
    """Batch size for video frame inference."""


@dataclass
class InferenceConfig:
    """Root configuration for inference engine.

    This is the main configuration object for inference.
    All sub-configurations have sensible defaults matching current behavior.
    """

    model_architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    """Model architecture configuration (frequency, spatial, classifier)."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    """Preprocessing pipeline configuration."""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    """Device configuration."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    """File paths configuration."""

    inference_options: InferenceOptionsConfig = field(default_factory=InferenceOptionsConfig)
    """Inference behavior options."""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'InferenceConfig':
        """Create InferenceConfig from dictionary (loaded from JSON).

        Recursively constructs nested dataclasses from nested dictionaries.

        Parameters:
            config_dict (dict): configuration dictionary
        Returns:
            InferenceConfig: constructed config object
        """
        def build_nested(datacls, data):
            """Recursively build nested dataclasses from dict."""
            if not isinstance(data, dict):
                return data

            if not hasattr(datacls, '__dataclass_fields__'):
                return data

            fieldtypes = {f.name: f.type for f in datacls.__dataclass_fields__.values()}
            kwargs = {}

            for key, value in data.items():
                if key in fieldtypes:
                    field_type = fieldtypes[key]

                    if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                        actual_types = [t for t in field_type.__args__ if t is not type(None)]
                        if actual_types:
                            field_type = actual_types[0]

                    if hasattr(field_type, '__dataclass_fields__'):
                        kwargs[key] = build_nested(field_type, value)
                    else:
                        kwargs[key] = value

            return datacls(**kwargs)

        return build_nested(cls, config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert InferenceConfig to dictionary for JSON serialization.
        Returns:
            dict: configuration as dictionary
        """
        return asdict(self)


__all__ = [
    'FFTConfig',
    'FFTConstantsConfig',
    'DCTConfig',
    'DCTConstantsConfig',
    'FusionMLPConfig',
    'FrequencyFeaturesConfig',
    'SpatialFeaturesConfig',
    'ClassifierConfig',
    'ModelArchitectureConfig',
    'FaceDetectionConfig',
    'FaceExtractionConfig',
    'FrameExtractionConfig',
    'PreprocessingConfig',
    'DeviceConfig',
    'PathsConfig',
    'InferenceOptionsConfig',
    'InferenceConfig',
]
