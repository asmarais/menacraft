import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging
import math
from .base_fft_extractor import BaseFFTExtractor


class TorchFFTExtractor(nn.Module, BaseFFTExtractor):
    """Extract FFT-based frequency features from face images to detect spectrum
    anomalies that indicate manipulation, upsampling, or generative artifacts."""

    def __init__(self,
        channel_mode: str,
        num_radial_bands: int,
        window_function: str,
        high_freq_emphasis: bool,
        feature_dim: int,
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
        """Initialize TorchFFTExtractor.
        Parameters:
            channel_mode (str): how to handle color channels
            num_radial_bands (int): number of radial frequency bands
            window_function (str): windowing function to reduce spectral leakage
            high_freq_emphasis (bool): emphasize high frequencies where deepfakes often have anomalies
            feature_dim (int): target output feature dimension
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
        nn.Module.__init__(self)
        BaseFFTExtractor.__init__(
            self,
            channel_mode=channel_mode,
            num_radial_bands=num_radial_bands,
            window_function=window_function,
            high_freq_emphasis=high_freq_emphasis,
            feature_dim=feature_dim,
            logger_name=("/".join(__file__.split("/")[-2:])),
            num_azimuthal_sectors=num_azimuthal_sectors,
            high_freq_threshold_ratio=high_freq_threshold_ratio,
            artifact_edge_width=artifact_edge_width,
            artifact_num_radial_samples=artifact_num_radial_samples,
            artifact_center_region_size=artifact_center_region_size,
            default_smoothing_kernel_size=default_smoothing_kernel_size,
            energy_band_start_multiplier=energy_band_start_multiplier,
            energy_band_end_multiplier=energy_band_end_multiplier,
            epsilon=epsilon
        )

        self.register_buffer('window_224', self._create_window(224))
        """Precomputed 224x224 window function"""

        self.logger.info("TorchFFTExtractor initialized successfully.")

    def _create_window(self, size: int) -> torch.Tensor:
        """Create 2D window function for spectral leakage reduction.
        Parameters:
            size (int): window size (assumed square)
        Returns:
            torch.Tensor: 2D window [size, size]
        """
        if self.window_function == "none":
            return torch.ones(size, size)

        #create 1D window
        if self.window_function == "hann":
            window_1d = torch.hann_window(size, periodic=False)
        elif self.window_function == "hamming":
            window_1d = torch.hamming_window(size, periodic=False)
        elif self.window_function == "blackman":
            window_1d = torch.blackman_window(size, periodic=False)
        else:
            window_1d = torch.ones(size)

        #create 2D window via outer product
        window_2d = torch.outer(window_1d, window_1d)
        return window_2d

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Extract FFT features from batch of images.
        Parameters:
            images (torch.Tensor): input images [batch, channels, height, width]
                - Expected: RGB images in PyTorch format (channels first)
                - Values in [0, 1] or [-1, 1]
        Returns:
            torch.Tensor: FFT features of shape [batch, feature_dim]
        Raises:
            ValueError: if image shape or format is invalid
        """
        try:
            assert len(images.shape) == 4, \
                f"Images must be 4D [batch, C, H, W], got shape {images.shape}"
            assert images.shape[1] == 3, \
                f"Images must have 3 channels (RGB), got {images.shape[1]}"

        except AssertionError as e:
            self.logger.error(f"- extract - {e}")
            raise ValueError(f"Invalid input images: {e}")

        batch_size = images.shape[0]

        #preprocess images to target channel mode
        preprocessed = self._preprocess_images(images)

        #apply FFT and compute magnitude spectrum
        magnitude_spectrum = self._apply_fft(preprocessed)

        #extract frequency features for entire batch
        raw_features = self._extract_frequency_features(magnitude_spectrum)

        #project to target dimension using learnable projection
        projected_features = self.projection(raw_features)

        self.logger.debug(f"- extract - Extracted FFT features with shape {projected_features.shape}")

        return projected_features

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to appropriate channel mode and apply windowing.
        Parameters:
            images (torch.Tensor): input images [batch, 3, height, width]
        Returns:
            torch.Tensor: preprocessed images with window applied [batch, height, width]
        Raises:
            ValueError: if image format is invalid
        """
        try:
            #convert to target channel mode
            if self.channel_mode == "luminance":
                channel_images = self._convert_to_luminance(images)
            elif self.channel_mode == "grayscale":
                channel_images = self._convert_to_grayscale(images)
            elif self.channel_mode == "per_channel":
                #per_channel mode not fully implemented
                self.logger.warning("- _preprocess_images - per_channel mode not fully supported, using luminance")
                channel_images = self._convert_to_luminance(images)
            else:
                raise ValueError(f"Unknown channel mode: {self.channel_mode}")

            #normalize to [0, 1] if not already
            if channel_images.max() > 1.0:
                channel_images = channel_images / 255.0

            #apply window function to reduce spectral leakage
            if self.window_function != "none":
                h, w = channel_images.shape[1], channel_images.shape[2]
                if h == 224 and w == 224:
                    #use precomputed window
                    window = self.window_224
                else:
                    #create window on-the-fly
                    window = self._create_window(h).to(channel_images.device)

                windowed_images = channel_images * window.unsqueeze(0)
            else:
                windowed_images = channel_images

            return windowed_images

        except Exception as e:
            self.logger.error(f"- _preprocess_images - {e}")
            raise ValueError(f"Error preprocessing images: {e}")

    def _convert_to_luminance(self, images: torch.Tensor) -> torch.Tensor:
        """Convert RGB images to luminance (Y channel from YCbCr).
        Parameters:
            images (torch.Tensor): RGB images [batch, 3, height, width]
        Returns:
            torch.Tensor: luminance images [batch, height, width]
        """
        #ITU-R BT.601 conversion weights
        r = images[:, 0, :, :]
        g = images[:, 1, :, :]
        b = images[:, 2, :, :]

        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return luminance

    def _convert_to_grayscale(self, images: torch.Tensor) -> torch.Tensor:
        """Convert RGB images to grayscale using average method.
        Parameters:
            images (torch.Tensor): RGB images [batch, 3, height, width]
        Returns:
            torch.Tensor: grayscale images [batch, height, width]
        """
        grayscale = torch.mean(images, dim=1)
        return grayscale

    def _apply_fft(self, images: torch.Tensor) -> torch.Tensor:
        """Apply 2D FFT and compute centered magnitude spectrum.
        Parameters:
            images (torch.Tensor): preprocessed images [batch, height, width]
        Returns:
            torch.Tensor: centered magnitude spectrum in log scale [batch, height, width]
        Raises:
            RuntimeError: if FFT computation fails
        """
        try:
            fft_result = torch.fft.fft2(images)

            #shift zero frequency to center
            fft_shifted = torch.fft.fftshift(fft_result)
            #compute magnitude spectrum
            magnitude = torch.abs(fft_shifted)

            #apply log scale for better dynamic range
            log_magnitude = torch.log(magnitude + 1.0)

            return log_magnitude

        except Exception as e:
            self.logger.error(f"- _apply_fft - Error computing FFT: {e}")
            raise RuntimeError(f"Error computing FFT: {e}")

    def _extract_frequency_features(self, magnitude_spectrum: torch.Tensor) -> torch.Tensor:
        """Extract comprehensive frequency features from magnitude spectrum.
        Parameters:
            magnitude_spectrum (torch.Tensor): centered magnitude spectrum [batch, height, width]
        Returns:
            torch.Tensor: raw frequency features [batch, num_raw_features]
        """
        batch_size, h, w = magnitude_spectrum.shape
        center = (h // 2, w // 2)

        radial_features = self._extract_radial_features(magnitude_spectrum, center)
        #directional features
        azimuthal_features = self._extract_azimuthal_features(magnitude_spectrum, center)

        #compute global spectral statistics
        global_features = self._compute_global_features(magnitude_spectrum, center)

        #deepfake indicators in high frequencies
        artifact_features = self._extract_artifact_features(magnitude_spectrum, center)

        #compute relationships between bands
        cross_band_features = self._compute_cross_band_features(radial_features)

        all_features = torch.cat([
            radial_features,      # num_radial_bands * 5 (mean, var, energy, peak, profile)
            azimuthal_features,   # 16 (8 directions × 2 stats)
            global_features,      # 8
            artifact_features,    # 6
            cross_band_features   # 5
        ], dim=1)

        return all_features

    def _extract_radial_features(self,
        magnitude_spectrum: torch.Tensor,
        center: Tuple[int, int]
    ) -> torch.Tensor:
        """Extract radial frequency band features.
        Parameters:
            magnitude_spectrum (torch.Tensor): magnitude spectrum [batch, height, width]
            center (Tuple[int, int]): center coordinates (cy, cx)
        Returns:
            torch.Tensor: radial features [batch, (num_radial_bands * 4) + num_radial_bands]
        """
        batch_size, h, w = magnitude_spectrum.shape
        cy, cx = center
        device = magnitude_spectrum.device

        # Maximum radius (distance to corner)
        max_radius = math.sqrt(cy**2 + cx**2)
        band_width = max_radius / self.num_radial_bands

        #create distance map
        y = torch.arange(h, device=device).view(-1, 1).float()
        x = torch.arange(w, device=device).view(1, -1).float()
        distance_map = torch.sqrt((y - cy)**2 + (x - cx)**2)

        # Collect features in lists to avoid in-place operations
        band_means_list = []
        band_variances_list = []
        band_energies_list = []
        band_peaks_list = []
        radial_profile_list = []

        for band_idx in range(self.num_radial_bands):
            # Define annular band
            inner_radius = band_idx * band_width
            outer_radius = (band_idx + 1) * band_width

            # Create mask [H, W]
            if band_idx == 0:
                mask = distance_map <= outer_radius
            else:
                mask = (distance_map > inner_radius) & (distance_map <= outer_radius)

            # Expand to batch dimension [B, H, W]
            mask_expanded = mask.unsqueeze(0).expand(batch_size, -1, -1)

            # Masked spectrum [B, H, W]
            masked_spectrum = magnitude_spectrum * mask_expanded

            # Count valid pixels per sample [B]
            counts = mask_expanded.sum(dim=[1, 2]).float()
            counts = torch.clamp(counts, min=1.0)  # Avoid division by zero

            # Vectorized statistics across batch
            band_mean = masked_spectrum.sum(dim=[1, 2]) / counts
            band_means_list.append(band_mean)

            # Variance (two-pass for numerical stability)
            mean_expanded = band_mean.view(batch_size, 1, 1)
            variance_sum = ((masked_spectrum - mean_expanded * mask_expanded) ** 2 * mask_expanded).sum(dim=[1, 2])
            band_variance = variance_sum / counts
            band_variances_list.append(band_variance)

            # Energy
            band_energy = (masked_spectrum ** 2).sum(dim=[1, 2])
            band_energies_list.append(band_energy)

            # Peaks (max with masking) - use torch.where to avoid in-place ops and CPU transfers
            neg_inf = torch.full_like(magnitude_spectrum, -float('inf'))
            masked_for_max = torch.where(mask_expanded, magnitude_spectrum, neg_inf)
            band_peak = masked_for_max.view(batch_size, -1).max(dim=1)[0]
            band_peak = torch.clamp(band_peak, min=0.0)
            band_peaks_list.append(band_peak)

            # Radial profile (same as mean)
            radial_profile_list.append(band_mean)

        # Stack into tensors [batch, num_bands]
        band_means = torch.stack(band_means_list, dim=1)
        band_variances = torch.stack(band_variances_list, dim=1)
        band_energies = torch.stack(band_energies_list, dim=1)
        band_peaks = torch.stack(band_peaks_list, dim=1)
        radial_profile = torch.stack(radial_profile_list, dim=1)

        # Concatenate: 4 statistics per band + radial profile
        features = torch.cat([
            band_means,
            band_variances,
            band_energies,
            band_peaks,
            radial_profile
        ], dim=1)

        return features

    def _extract_azimuthal_features(self,
        magnitude_spectrum: torch.Tensor,
        center: Tuple[int, int]
    ) -> torch.Tensor:
        """Extract directional (azimuthal) frequency features.
        Parameters:
            magnitude_spectrum (torch.Tensor): magnitude spectrum [batch, height, width]
            center (Tuple[int, int]): center coordinates
        Returns:
            torch.Tensor: azimuthal features [batch, 16] (8 directions x 2 stats)
        """
        batch_size, h, w = magnitude_spectrum.shape
        cy, cx = center
        device = magnitude_spectrum.device

        #center relative coordinates grid
        y = torch.arange(h, device=device).view(-1, 1).float()
        x = torch.arange(w, device=device).view(1, -1).float()
        y_rel = y - cy
        x_rel = x - cx

        angles = torch.atan2(y_rel, x_rel)

        sector_width = self._get_sector_width()

        # Collect features in lists to avoid in-place operations
        sector_means_list = []
        sector_variances_list = []

        for sector_idx in range(self.NUM_AZIMUTHAL_SECTORS):
            angle_start = -math.pi + sector_idx * sector_width
            angle_end = angle_start + sector_width

            # Create mask [H, W]
            if sector_idx == self.NUM_AZIMUTHAL_SECTORS - 1:
                mask = (angles >= angle_start) | (angles < -math.pi + sector_width)
            else:
                mask = (angles >= angle_start) & (angles < angle_end)

            # Expand to batch dimension [B, H, W]
            mask_expanded = mask.unsqueeze(0).expand(batch_size, -1, -1)

            # Masked spectrum [B, H, W]
            masked_spectrum = magnitude_spectrum * mask_expanded

            # Count valid pixels per sample [B]
            counts = mask_expanded.sum(dim=[1, 2]).float()
            counts = torch.clamp(counts, min=1.0)  # Avoid division by zero

            # Vectorized statistics across batch
            sector_mean = masked_spectrum.sum(dim=[1, 2]) / counts
            sector_means_list.append(sector_mean)

            # Variance (two-pass for numerical stability)
            mean_expanded = sector_mean.view(batch_size, 1, 1)
            variance_sum = ((masked_spectrum - mean_expanded * mask_expanded) ** 2 * mask_expanded).sum(dim=[1, 2])
            sector_variance = variance_sum / counts
            sector_variances_list.append(sector_variance)

        # Stack into tensors [batch, num_sectors]
        sector_means = torch.stack(sector_means_list, dim=1)
        sector_variances = torch.stack(sector_variances_list, dim=1)

        features = torch.cat([sector_means, sector_variances], dim=1)
        return features

    def _compute_global_features(self,
        magnitude_spectrum: torch.Tensor,
        center: Tuple[int, int]
    ) -> torch.Tensor:
        """Compute global spectral statistics.
        Parameters:
            magnitude_spectrum (torch.Tensor): magnitude spectrum [batch, height, width]
            center (Tuple[int, int]): center coordinates
        Returns:
            torch.Tensor: global features [batch, 8]
        """
        batch_size, h, w = magnitude_spectrum.shape
        device = magnitude_spectrum.device

        #DC magnitude (center value)
        dc_magnitude = magnitude_spectrum[:, center[0], center[1]]

        #total energy
        total_energy = torch.sum(magnitude_spectrum ** 2, dim=[1, 2])

        #high-frequency energy ratio
        high_freq_ratio = self._compute_high_freq_ratio(magnitude_spectrum, center)

        #spectral centroid (weighted mean frequency)
        y = torch.arange(h, device=device).view(-1, 1, 1).float()
        x = torch.arange(w, device=device).view(1, -1, 1).float()

        weights = magnitude_spectrum.permute(1, 2, 0)
        weighted_y = torch.sum(y * weights, dim=[0, 1]) / (torch.sum(weights, dim=[0, 1]) + 1e-10)
        weighted_x = torch.sum(x * weights, dim=[0, 1]) / (torch.sum(weights, dim=[0, 1]) + 1e-10)
        centroid_dist = torch.sqrt((weighted_y - center[0])**2 + (weighted_x - center[1])**2)

        #spectral rolloff (frequency below which 85% of energy is contained)
        cumsum_energy = torch.cumsum(magnitude_spectrum.flatten(1) ** 2, dim=1)
        rolloff_threshold = 0.85 * cumsum_energy[:, -1]
        rolloff_idx = torch.searchsorted(cumsum_energy, rolloff_threshold.unsqueeze(1)).squeeze(1).float()

        #spectral flatness (measure of noise vs tones)
        eps = 1e-10
        geometric_mean = torch.exp(torch.mean(torch.log(magnitude_spectrum + eps), dim=[1, 2]))
        arithmetic_mean = torch.mean(magnitude_spectrum, dim=[1, 2])
        flatness = geometric_mean / (arithmetic_mean + eps)

        #spectral entropy
        prob = magnitude_spectrum / (torch.sum(magnitude_spectrum, dim=[1, 2], keepdim=True) + eps)
        entropy = -torch.sum(prob * torch.log2(prob + eps), dim=[1, 2])

        #peak frequency magnitude
        peak_freq = torch.max(magnitude_spectrum.flatten(1), dim=1)[0]

        #stack all features
        features = torch.stack([
            dc_magnitude,
            total_energy,
            high_freq_ratio,
            centroid_dist,
            rolloff_idx,
            flatness,
            entropy,
            peak_freq
        ], dim=1)

        return features

    def _compute_high_freq_ratio(self,
        magnitude_spectrum: torch.Tensor,
        center: Tuple[int, int]
    ) -> torch.Tensor:
        """Compute ratio of high-frequency energy to total energy.
        Parameters:
            magnitude_spectrum (torch.Tensor): magnitude spectrum [batch, height, width]
            center (Tuple[int, int]): center coordinates
        Returns:
            torch.Tensor: high frequency energy ratio [batch]
        """
        batch_size, h, w = magnitude_spectrum.shape
        cy, cx = center
        device = magnitude_spectrum.device

        max_radius = self._compute_max_radius(center)
        high_freq_threshold = max_radius * self.HIGH_FREQ_THRESHOLD_RATIO

        #create distance map
        y = torch.arange(h, device=device).view(-1, 1).float()
        x = torch.arange(w, device=device).view(1, -1).float()
        distance_map = torch.sqrt((y - cy)**2 + (x - cx)**2)

        #create mask for high frequencies
        high_freq_mask = distance_map > high_freq_threshold

        high_freq_energy = torch.sum(magnitude_spectrum * high_freq_mask.unsqueeze(0) ** 2, dim=[1, 2])
        total_energy = torch.sum(magnitude_spectrum ** 2, dim=[1, 2])

        return high_freq_energy / (total_energy + self.EPSILON)

    def _extract_artifact_features(self,
        magnitude_spectrum: torch.Tensor,
        center: Tuple[int, int]
    ) -> torch.Tensor:
        """Extract high-frequency artifact features specific to deepfakes.
        Parameters:
            magnitude_spectrum (torch.Tensor): magnitude spectrum [batch, height, width]
            center (Tuple[int, int]): center coordinates
        Returns:
            torch.Tensor: artifact features [batch, 6]
        """
        batch_size, h, w = magnitude_spectrum.shape
        device = magnitude_spectrum.device

        #Nyquist frequencies on edges of the spectrum can indicate checkerboard artifacts
        edge = self.ARTIFACT_EDGE_WIDTH
        top_edge = torch.mean(magnitude_spectrum[:, 0:edge, :], dim=[1, 2])
        bottom_edge = torch.mean(magnitude_spectrum[:, -edge:, :], dim=[1, 2])
        left_edge = torch.mean(magnitude_spectrum[:, :, 0:edge], dim=[1, 2])
        right_edge = torch.mean(magnitude_spectrum[:, :, -edge:], dim=[1, 2])
        checkerboard_score = (top_edge + bottom_edge + left_edge + right_edge) / 4.0

        #upsampling signatures (periodic peaks in spectrum)
        max_radius = self._compute_max_radius(center)
        num_samples = self.ARTIFACT_NUM_RADIAL_SAMPLES
        radii = torch.linspace(0, max_radius, num_samples, device=device)

        #create distance map
        y = torch.arange(h, device=device).view(-1, 1).float()
        x = torch.arange(w, device=device).view(1, -1).float()
        distance_map = torch.sqrt((y - center[0])**2 + (x - center[1])**2)

        # Vectorized radial profile sampling - collect in list to avoid in-place ops
        radial_profile_samples_list = []
        for i, r in enumerate(radii):
            # Create mask [H, W]
            mask = (distance_map >= r - 1) & (distance_map < r + 1)

            # Expand to batch dimension [B, H, W]
            mask_expanded = mask.unsqueeze(0).expand(batch_size, -1, -1)

            # Masked spectrum [B, H, W]
            masked_spectrum = magnitude_spectrum * mask_expanded

            # Count valid pixels per sample [B]
            counts = mask_expanded.sum(dim=[1, 2]).float().clamp(min=1.0)

            # Vectorized mean across batch
            profile_sample = masked_spectrum.sum(dim=[1, 2]) / counts
            radial_profile_samples_list.append(profile_sample)

        radial_profile_samples = torch.stack(radial_profile_samples_list, dim=1)
        upsampling_score = torch.var(radial_profile_samples, dim=1)

        #asymmetry scores (horizontal, vertical)
        left_half = magnitude_spectrum[:, :, :w//2]
        right_half = magnitude_spectrum[:, :, w//2:]
        horizontal_asymmetry = torch.abs(torch.mean(left_half, dim=[1, 2]) - torch.mean(right_half, dim=[1, 2]))

        top_half = magnitude_spectrum[:, :h//2, :]
        bottom_half = magnitude_spectrum[:, h//2:, :]
        vertical_asymmetry = torch.abs(torch.mean(top_half, dim=[1, 2]) - torch.mean(bottom_half, dim=[1, 2]))

        #coherence of medium-high frequency region
        region_size = self.ARTIFACT_CENTER_REGION_SIZE
        center_region = magnitude_spectrum[:, center[0]-region_size:center[0]+region_size, center[1]-region_size:center[1]+region_size]
        high_freq_variance = torch.var(center_region, dim=[1, 2])

        #spectral irregularity
        smoothed = self._smooth_spectrum(magnitude_spectrum)
        irregularity = torch.mean(torch.abs(magnitude_spectrum - smoothed), dim=[1, 2])

        features = torch.stack([
            checkerboard_score,
            upsampling_score,
            horizontal_asymmetry,
            vertical_asymmetry,
            high_freq_variance,
            irregularity
        ], dim=1)

        return features

    def _smooth_spectrum(self, spectrum: torch.Tensor, kernel_size: int = None) -> torch.Tensor:
        """Smooth spectrum using average pooling.
        Parameters:
            spectrum (torch.Tensor): input spectrum [batch, height, width]
            kernel_size (int): smoothing kernel size
        Returns:
            torch.Tensor: smoothed spectrum [batch, height, width]
        """
        if kernel_size is None:
            kernel_size = self.DEFAULT_SMOOTHING_KERNEL_SIZE

        #use average pooling for smoothing
        spectrum_4d = spectrum.unsqueeze(1)

        #apply average pooling with same padding
        padding = kernel_size // 2
        smoothed = F.avg_pool2d(spectrum_4d, kernel_size, stride=1, padding=padding)

        return smoothed.squeeze(1)

    def _compute_cross_band_features(self, radial_features: torch.Tensor) -> torch.Tensor:
        """Compute cross-band interaction features.
        Parameters:
            radial_features (torch.Tensor): radial features [batch, num_bands * 5]
        Returns:
            torch.Tensor: cross-band features [batch, 5]
        """
        batch_size = radial_features.shape[0]
        device = radial_features.device

        #band energies from radial features [means, variances, energies, peaks, profile]
        num_bands = self.num_radial_bands
        energies = radial_features[:, num_bands * self.ENERGY_BAND_START_MULTIPLIER: num_bands * self.ENERGY_BAND_END_MULTIPLIER]

        if num_bands < 2:
            return torch.zeros(batch_size, 5, device=device)

        #energy ratios between consecutive bands
        ratio_1_2 = energies[:, 1] / (energies[:, 0] + self.EPSILON)
        ratio_2_3 = energies[:, 2] / (energies[:, 1] + self.EPSILON) if num_bands > 2 else torch.zeros(batch_size, device=device)
        ratio_low_high = energies[:, 0] / (energies[:, -1] + self.EPSILON)

        energy_slope = (energies[:, -1] - energies[:, 0]) / (num_bands + self.EPSILON)

        #entropy-like measure of energy distribution
        total_energy = torch.sum(energies, dim=1, keepdim=True) + self.EPSILON
        energy_dist = energies / total_energy
        energy_concentration = -torch.sum(energy_dist * torch.log2(energy_dist + self.EPSILON), dim=1)

        features = torch.stack([
            ratio_1_2,
            ratio_2_3,
            ratio_low_high,
            energy_slope,
            energy_concentration
        ], dim=1)

        return features
