from frequal.base_dct_extractor import BaseDCTExtractor
from frequal.base_fft_extractor import BaseFFTExtractor
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .np_dct_extractor import NpDCTExtractor
from .np_fft_extractor import NpFFTExtractor
from .torch_dct_extractor import TorchDCTExtractor
from .torch_fft_extractor import TorchFFTExtractor
from .fusion_mlp import FusionMLP
from .utils import torch_to_numpy


class FrequencyFeatureExtractor(nn.Module):
    """PyTorch module encapsulating combined DCT and FFT frequency feature extraction."""

    def __init__(self,
        fft_feature_dim: int =512,
        fft_num_radial_bands: int =8,
        fft_window_function: str ="hann",
        fft_high_freq_emphasis: bool =True,
        dct_feature_dim: int =512,
        dct_block_size: int =8,
        channel_mode: str ="luminance",
        dct_aggregation_method: str ="frequency_bands",
        dct_num_frequency_bands: int =4,
        fusion_output_dim: Optional[int] =None,
        fusion_hidden_dims: Optional[List[int]] =None,
        fusion_dropout_rate: float =0.3,
        fusion_use_batch_norm: bool =True,
        path_to_weights: Optional[str] =None,
        use_gpu_extractors: bool =True,
        fft_num_azimuthal_sectors: int =8,
        fft_high_freq_threshold_ratio: float =0.5,
        fft_artifact_edge_width: int =5,
        fft_artifact_num_radial_samples: int =50,
        fft_artifact_center_region_size: int =20,
        fft_default_smoothing_kernel_size: int =5,
        fft_energy_band_start_multiplier: int =2,
        fft_energy_band_end_multiplier: int =3,
        fft_epsilon: float =1e-10,
        dct_epsilon: float =1e-10,
    ):
        """Initialize FrequencyFeatureExtractor.
        Parameters:
            fft_feature_dim (int): FFT feature dimension
            fft_num_radial_bands (int): number of concentric rings that divide the spectrum
            fft_window_function (str): window function to apply to FFT
            fft_high_freq_emphasis (bool): whether to emphasize high frequencies
            dct_feature_dim (int): DCT feature dimension
            dct_block_size (int): DCT block size in pixels
            channel_mode (str): image channel processing mode
            dct_aggregation_method (str): DCT aggregation method
            dct_num_frequency_bands (int): number of frequency bands for DCT aggregation
            fusion_output_dim (int, optional): output dimension after MLP fusion
            fusion_hidden_dims (List[int], optional): hidden layer dimensions for fusion MLP
            fusion_dropout_rate (float): dropout rate for fusion MLP
            fusion_use_batch_norm (bool): whether to use batch normalization in fusion MLP
            path_to_weights (str, optional): path to pretrained MLP weights
            use_gpu_extractors (bool): whether to use GPU for extraction
            fft_num_azimuthal_sectors (int): number of angular sectors for FFT
            fft_high_freq_threshold_ratio (float): high frequency threshold ratio for FFT
            fft_artifact_edge_width (int): edge width for FFT artifact detection
            fft_artifact_num_radial_samples (int): radial samples for FFT artifact detection
            fft_artifact_center_region_size (int): center region size for FFT analysis
            fft_default_smoothing_kernel_size (int): smoothing kernel size for FFT
            fft_energy_band_start_multiplier (int): energy band start multiplier for FFT
            fft_energy_band_end_multiplier (int): energy band end multiplier for FFT
            fft_epsilon (float): epsilon for FFT numerical stability
            dct_epsilon (float): epsilon for DCT numerical stability
        Raises:
            ValueError: if parameters are invalid
            FileNotFoundError: if weights file doesn't exist
        """
        super(FrequencyFeatureExtractor, self).__init__()

        self.logger: logging.Logger = logging.getLogger("frequal.FrequencyFeatureExtractor")
        """Logger instance for the FrequencyFeatureExtractor class"""

        try:
            assert dct_feature_dim > 0, f"DCT feature dimension must be positive, got {dct_feature_dim}"
            assert fft_feature_dim > 0, f"FFT feature dimension must be positive, got {fft_feature_dim}"
            assert dct_block_size in [8, 16], f"DCT block size must be 8 or 16, got {dct_block_size}"
            assert fft_num_radial_bands > 0, f"FFT radial bands must be positive, got {fft_num_radial_bands}"
            
        except AssertionError as e:
            self.logger.fatal(f"Invalid parameters: {e}")
            raise ValueError(f"Invalid parameters: {e}")

        self.dct_feature_dim: int = dct_feature_dim
        """DCT feature dimension"""
        self.fft_feature_dim: int = fft_feature_dim
        """FFT feature dimension"""
        self.combined_dim: int = dct_feature_dim + fft_feature_dim
        """Combined feature dimension (DCT + FFT)"""
        self.fusion_output_dim: int = fusion_output_dim if fusion_output_dim is not None else self.combined_dim
        """Output dimension after MLP fusion"""
        self.use_gpu_extractors: bool = use_gpu_extractors
        """Whether to use GPU-based PyTorch extractors or CPU-based NumPy extractors"""

        self.dct_extractor: BaseDCTExtractor
        """DCT PyTorch or Numpy based feature extractor"""
        self.fft_extractor: BaseFFTExtractor
        """FFT PyTorch or Numpy based feature extractor"""
        if use_gpu_extractors:
            self.dct_extractor = TorchDCTExtractor(
                block_size=dct_block_size,
                channel_mode=channel_mode,
                aggregation_method=dct_aggregation_method,
                num_frequency_bands=dct_num_frequency_bands,
                feature_dim=dct_feature_dim,
                epsilon=dct_epsilon
            )

            self.fft_extractor = TorchFFTExtractor(
                channel_mode=channel_mode,
                num_radial_bands=fft_num_radial_bands,
                window_function=fft_window_function,
                high_freq_emphasis=fft_high_freq_emphasis,
                feature_dim=fft_feature_dim,
                num_azimuthal_sectors=fft_num_azimuthal_sectors,
                high_freq_threshold_ratio=fft_high_freq_threshold_ratio,
                artifact_edge_width=fft_artifact_edge_width,
                artifact_num_radial_samples=fft_artifact_num_radial_samples,
                artifact_center_region_size=fft_artifact_center_region_size,
                default_smoothing_kernel_size=fft_default_smoothing_kernel_size,
                energy_band_start_multiplier=fft_energy_band_start_multiplier,
                energy_band_end_multiplier=fft_energy_band_end_multiplier,
                epsilon=fft_epsilon
            )
        else:
            self.dct_extractor = NpDCTExtractor(
                block_size=dct_block_size,
                channel_mode=channel_mode,
                aggregation_method=dct_aggregation_method,
                num_frequency_bands=dct_num_frequency_bands,
                feature_dim=dct_feature_dim,
                epsilon=dct_epsilon
            )

            self.fft_extractor = NpFFTExtractor(
                channel_mode=channel_mode,
                num_radial_bands=fft_num_radial_bands,
                window_function=fft_window_function,
                high_freq_emphasis=fft_high_freq_emphasis,
                feature_dim=fft_feature_dim,
                num_azimuthal_sectors=fft_num_azimuthal_sectors,
                high_freq_threshold_ratio=fft_high_freq_threshold_ratio,
                artifact_edge_width=fft_artifact_edge_width,
                artifact_num_radial_samples=fft_artifact_num_radial_samples,
                artifact_center_region_size=fft_artifact_center_region_size,
                default_smoothing_kernel_size=fft_default_smoothing_kernel_size,
                energy_band_start_multiplier=fft_energy_band_start_multiplier,
                energy_band_end_multiplier=fft_energy_band_end_multiplier,
                epsilon=fft_epsilon
            )

        self.fusion_mlp: FusionMLP = FusionMLP(
            input_dim=self.combined_dim,
            output_dim=self.fusion_output_dim,
            hidden_dims=fusion_hidden_dims,
            dropout_rate=fusion_dropout_rate,
            use_batch_norm=fusion_use_batch_norm
        )
        """MLP for feature fusion"""

        # Initialize DCT and FFT projection layers
        self._initialize_projections()

        #load pretrained weights if provided
        if path_to_weights is not None:
            self.load_weights(path_to_weights)
            self.logger.info(f"Loaded pretrained weights from {path_to_weights}")

        self.logger.info("FrequencyFeatureExtractor initialized successfully.")

    def _initialize_projections(self) -> None:
        """Initialize DCT and FFT projection layers with Xavier initialization.

        Xavier initialization is appropriate for projection layers that don't use ReLU activation.
        """
        for extractor in [self.dct_extractor, self.fft_extractor]:
            if hasattr(extractor, 'projection'):
                for m in extractor.projection.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        self.logger.info("DCT/FFT projections initialized with Xavier initialization")

    def forward(self, x: torch.Tensor, parallelize: bool =False) -> torch.Tensor:
        """Extract combined frequency features from input images.
        Parameters:
            x (torch.Tensor): input images [batch, 3, 224, 224]
                - Expected: RGB images in PyTorch format (channels first)
                - Values normalized to [0, 1] or [-1, 1]
                - Will be converted to [0, 255] BGR format for frequency analysis
            parallelize (bool): whether to parallelize DCT and FFT extraction.
                Only used when use_gpu_extractors=False (CPU mode with NumPy).
                GPU extractors are already parallelized by CUDA.
        Returns:
            torch.Tensor: fused frequency features [batch, fusion_output_dim]
                where fusion_output_dim is the output dimension of the MLP fusion layer
        Raises:
            ValueError: if input shape is invalid
        """
        try:
            assert len(x.shape) == 4, f"Input must be 4D tensor [batch, channels, height, width], got shape {x.shape}"
            assert x.shape[1] == 3, f"Input must have 3 channels (RGB), got {x.shape[1]}"
            assert x.shape[2] == 224 and x.shape[3] == 224, f"Input must be 224x224, got {x.shape[2]}x{x.shape[3]}"

        except AssertionError as e:
            self.logger.error(f"- forward - {e}")
            raise ValueError(f"Invalid input shape: {e}")

        if self.use_gpu_extractors:
            #use PyTorch extractors directly on GPU tensors
            #GPU operations are already parallelized, no need for threading
            dct_features = self.dct_extractor.extract(x)
            fft_features = self.fft_extractor.extract(x)

            #combine features (already PyTorch tensors)
            combined_tensor = torch.cat([dct_features, fft_features], dim=1)
        else:
            #convert to NumPy for CPU-based extractors
            x_numpy = torch_to_numpy(x)

            if parallelize:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    dct_thread = executor.submit(self.dct_extractor.extract, x_numpy)
                    fft_thread = executor.submit(self.fft_extractor.extract, x_numpy)
                    dct_features = dct_thread.result()
                    fft_features = fft_thread.result()
            else:
                dct_features = self.dct_extractor.extract(x_numpy)
                fft_features = self.fft_extractor.extract(x_numpy)

            #combine features (NumPy arrays)
            combined_features = np.concatenate([dct_features, fft_features], axis=1)

            #convert back to PyTorch tensor
            combined_tensor = torch.from_numpy(combined_features).float()
            combined_tensor = combined_tensor.to(x.device)

        #fuse features using MLP
        fused_features = self.fusion_mlp(combined_tensor)

        self.logger.debug(f"- forward - Extracted frequency features with shape {fused_features.shape} (parallel={parallelize})")

        return fused_features

    def save_weights(self, output_path: str) -> None:
        """Save learnable MLP weights to disk. Includes the fusion MLP
        and the projection layers within DCT and FFT extractors.
        Parameters:
            output_path (str): output file path
        Raises:
            RuntimeError: if saving fails
        """
        try:
            path_obj = Path(output_path)
            #create parent directory if it doesn't exist
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            state_dict = {
                'fusion_mlp': self.fusion_mlp.state_dict(),
                'dct_projection': self.dct_extractor.projection.state_dict(),
                'fft_projection': self.fft_extractor.projection.state_dict(),
            }

            torch.save(state_dict, path_obj)
            self.logger.info(f"- save_weights - Saved weights to {path_obj}")

        except Exception as e:
            self.logger.error(f"- save_weights - Error saving weights: {e}")
            raise RuntimeError(f"Error saving weights: {e}")

    def load_weights(self, input_path: str) -> None:
        """Load pretrained MLP weights from disk.
        Parameters:
            input_path (str): path to weights file (.pth)
        Raises:
            ValueError: if parameters are invalid
            RuntimeError: if loading fails
        """
        try:
            path = Path(input_path)
            assert path.exists(), f"Weights file does not exist: {input_path}"

        except AssertionError as e:
            self.logger.error(f"- load_weights - {e}")
            raise ValueError(f"Invalid parameter: {e}")

        try:
            state_dict: dict[str, torch.Tensor] = torch.load(path, map_location='cpu')

            #load fusion MLP (supports both old 'fusion_projection' and new 'fusion_mlp' keys for backward compatibility)
            if 'fusion_mlp' in state_dict:
                self.fusion_mlp.load_state_dict(state_dict['fusion_mlp'])
            elif 'fusion_projection' in state_dict:
                self.logger.warning("- load_weights - Loading old 'fusion_projection' weights. Consider resaving with new format.")
                #attempt to load old linear projection into first layer of MLP if compatible
                try:
                    old_weights = state_dict['fusion_projection']
                    #check if dimensions match first linear layer
                    first_linear = self.fusion_mlp.mlp[0]
                    if isinstance(first_linear, nn.Linear):
                        if first_linear.weight.shape == old_weights['weight'].shape:
                            first_linear.load_state_dict(old_weights)
                            self.logger.info("- load_weights - Successfully loaded old projection weights into MLP first layer")
                except Exception as e:
                    self.logger.warning(f"- load_weights - Could not load old projection weights: {e}")

            if 'dct_projection' in state_dict:
                self.dct_extractor.projection.load_state_dict(state_dict['dct_projection'])

            if 'fft_projection' in state_dict:
                self.fft_extractor.projection.load_state_dict(state_dict['fft_projection'])

            self.logger.info(f"- load_weights - Loaded weights from {path}")

        except Exception as e:
            self.logger.error(f"- load_weights - Error loading weights: {e}")
            raise RuntimeError(f"Error loading weights: {e}")
