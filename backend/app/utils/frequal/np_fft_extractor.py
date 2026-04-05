import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import logging
from .utils import convert_to_luminance, convert_to_grayscale, convert_per_channel, create_2d_window, create_radial_mask, compute_spectral_stats, format_image
from .base_fft_extractor import BaseFFTExtractor

class NpFFTExtractor(BaseFFTExtractor):
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
        """Initialize NpFFTExtractor.
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
        super().__init__(
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

        self.logger.info("FFTExtractor initialized successfully.")

    def extract(self, images: np.ndarray) -> np.ndarray:
        """Extract FFT frequency features from batch of images.
        Parameters:
            images (np.ndarray): input images with shape [batch, height, width, channels]
                or [batch, height, width] for grayscale
        Returns:
            np.ndarray: frequency features of shape [batch, feature_dim]
        Raises:
            ValueError: if parameters are invalid
            RuntimeError: if extraction fails
        """
        try:
            assert images is not None and images.size > 0, "Empty or None images"
            assert len(images.shape) in [3, 4], f"Images must be 3D [batch, H, W] or 4D [batch, H, W, C], got shape {images.shape}"

        except AssertionError as e:
            self.logger.error(f"- extract - {e}")
            raise ValueError(f"Invalid parameters: {e}")

        batch_size = images.shape[0]
        batch_features = []

        for i in range(batch_size):
            image = images[i]

            try:
                preprocessed = self._preprocess_image(image)
                magnitude_spectrum = self._apply_fft(preprocessed)
                raw_features = self._extract_frequency_features(magnitude_spectrum)

                features_tensor = torch.from_numpy(raw_features).float().unsqueeze(0)
                projected_features = self.projection(features_tensor)
                features = projected_features.detach().numpy().squeeze(0)

                batch_features.append(features)

            except Exception as e:
                self.logger.error(f"- extract - Error processing image {i}: {e}")
                raise RuntimeError(f"Error extracting features from image {i}: {e}")

        result = np.stack(batch_features, axis=0)
        self.logger.debug(f"- extract - Extracted features with shape {result.shape}")
        return result

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to appropriate channel mode and apply windowing.
        Parameters:
            image (np.ndarray): input image [height, width, channels] or [height, width]
        Returns:
            np.ndarray: preprocessed image with window applied [height, width]
        Raises:
            ValueError: if image format is invalid
        """
        try:
            image = format_image(image.copy())

            match self.channel_mode:
                case "luminance":
                    channel_image = convert_to_luminance(image)
                case "grayscale":
                    channel_image = convert_to_grayscale(image)
                case "per_channel":
                    channel_image = convert_per_channel(image)
                case _:
                    raise ValueError(f"Unknown channel mode: {self.channel_mode}")

            image = channel_image.astype(np.float32)

            if self.window_function is not None:
                window = create_2d_window(image.shape, self.window_function)
                image *= window

            return image

        except Exception as e:
            self.logger.error(f"- _preprocess_image - {e}")
            raise ValueError(f"Error preprocessing image: {e}")

    def _apply_fft(self, image: np.ndarray) -> np.ndarray:
        """Apply 2D FFT and compute centered magnitude spectrum.
        Parameters:
            image (np.ndarray): preprocessed image [height, width]
        Returns:
            np.ndarray: centered magnitude spectrum in log scale [height, width]
        Raises:
            RuntimeError: if FFT computation fails
        """
        try:
            fft_result = np.fft.fft2(image)

            #shift zero frequency to center
            fft_shifted = np.fft.fftshift(fft_result)
            #compute magnitude spectrum
            magnitude = np.abs(fft_shifted)

            #apply log scale for better dynamic range
            log_magnitude = np.log(magnitude + 1.0)

            return log_magnitude

        except Exception as e:
            self.logger.error(f"- _apply_fft - Error computing FFT: {e}")
            raise RuntimeError(f"Error computing FFT: {e}")

    def _extract_frequency_features(self, magnitude_spectrum: np.ndarray) -> np.ndarray:
        """Extract comprehensive frequency features from magnitude spectrum.
        Parameters:
            magnitude_spectrum (np.ndarray): centered magnitude spectrum [height, width]
        Returns:
            np.ndarray: raw frequency features [num_raw_features]
        """
        h, w = magnitude_spectrum.shape
        center = (h // 2, w // 2)

        radial_features = self._extract_radial_features(magnitude_spectrum, center)
        #directional features
        azimuthal_features = self._extract_azimuthal_features(magnitude_spectrum, center)

        #compute global spectral statistics
        spectral_stats = compute_spectral_stats(magnitude_spectrum)
        global_features = np.array([
            spectral_stats["centroid"],
            spectral_stats["rolloff"],
            spectral_stats["flatness"],
            spectral_stats["entropy"],
            spectral_stats["peak_freq"]
        ])
        dc_magnitude = magnitude_spectrum[center[0], center[1]]
        total_energy = np.sum(magnitude_spectrum ** 2)
        high_freq_energy_ratio = self._compute_high_freq_ratio(magnitude_spectrum, center)
        additional_global = np.array([dc_magnitude, total_energy, high_freq_energy_ratio])
        global_features = np.concatenate([global_features, additional_global])

        #deepfake indicators in high frequencies
        artifact_features = self._extract_artifact_features(magnitude_spectrum, center)

        #compute relationships between bands
        cross_band_features = self._compute_cross_band_features(radial_features)

        all_features = np.concatenate([
            radial_features,      # num_radial_bands * 5 (mean, var, energy, peak, profile)
            azimuthal_features,   # 16 (8 directions × 2 stats)
            global_features,      # 8
            artifact_features,    # 6
            cross_band_features   # 5
        ])

        return all_features.astype(np.float32)

    def _extract_radial_features(self,
        magnitude_spectrum: np.ndarray,
        center: Tuple[int, int]
    ) -> np.ndarray:
        """Extract radial frequency band features.
        Parameters:
            magnitude_spectrum (np.ndarray): magnitude spectrum
            center (Tuple[int, int]): center coordinates (cy, cx)
        Returns:
            np.ndarray: radial features [(num_radial_bands * 4) + num_radial_bands]
        """
        h, w = magnitude_spectrum.shape
        cy, cx = center

        # Maximum radius (distance to corner)
        max_radius = np.sqrt(cy**2 + cx**2)
        band_width = max_radius / self.num_radial_bands

        band_means = []
        band_variances = []
        band_energies = []
        band_peaks = []
        radial_profile = []

        for band_idx in range(self.num_radial_bands):
            # Define annular band
            inner_radius = band_idx * band_width
            outer_radius = (band_idx + 1) * band_width

            # Create masks for inner and outer circles
            if band_idx == 0:
                mask = create_radial_mask(center, (h, w), outer_radius)
            else:
                outer_mask = create_radial_mask(center, (h, w), outer_radius)
                inner_mask = create_radial_mask(center, (h, w), inner_radius)
                mask = outer_mask & ~inner_mask

            # Extract values in this band
            band_values = magnitude_spectrum[mask]

            if len(band_values) > 0:
                band_means.append(np.mean(band_values))
                band_variances.append(np.var(band_values))
                band_energies.append(np.sum(band_values ** 2))
                band_peaks.append(np.max(band_values))
                radial_profile.append(np.mean(band_values))
            else:
                band_means.append(0.0)
                band_variances.append(0.0)
                band_energies.append(0.0)
                band_peaks.append(0.0)
                radial_profile.append(0.0)

        # Concatenate: 4 statistics per band + radial profile
        features = np.concatenate([
            band_means,
            band_variances,
            band_energies,
            band_peaks,
            radial_profile
        ])

        return np.array(features, dtype=np.float32)

    def _extract_azimuthal_features(self,
        magnitude_spectrum: np.ndarray,
        center: Tuple[int, int]
    ) -> np.ndarray:
        """Extract directional (azimuthal) frequency features.
        Parameters:
            magnitude_spectrum (np.ndarray): magnitude spectrum
            center (Tuple[int, int]): center coordinates
        Returns:
            np.ndarray: azimuthal features [16] (8 directions x 2 stats)
        """
        h, w = magnitude_spectrum.shape
        cy, cx = center

        #center relative coordinates grid
        y, x = np.ogrid[:h, :w]
        y_rel = y - cy
        x_rel = x - cx

        angles = np.arctan2(y_rel, x_rel)

        sector_width = self._get_sector_width()

        sector_means = []
        sector_variances = []

        for sector_idx in range(self.NUM_AZIMUTHAL_SECTORS):
            angle_start = -np.pi + sector_idx * sector_width
            angle_end = angle_start + sector_width

            if sector_idx == self.NUM_AZIMUTHAL_SECTORS - 1:
                mask = (angles >= angle_start) | (angles < -np.pi + sector_width)
            else:
                mask = (angles >= angle_start) & (angles < angle_end)

            #apply mask to magnitude spectrum to extract sector values
            sector_values = magnitude_spectrum[mask]

            if len(sector_values) > 0:
                sector_means.append(np.mean(sector_values))
                sector_variances.append(np.var(sector_values))
            else:
                sector_means.append(0.0)
                sector_variances.append(0.0)

        features = np.concatenate([sector_means, sector_variances])
        return np.array(features, dtype=np.float32)

    def _compute_high_freq_ratio(self,
        magnitude_spectrum: np.ndarray,
        center: Tuple[int, int]
    ) -> float:
        """Compute ratio of high-frequency energy to total energy.
        Parameters:
            magnitude_spectrum (np.ndarray): magnitude spectrum
            center (Tuple[int, int]): center coordinates
        Returns:
            float: high frequency energy ratio
        """
        h, w = magnitude_spectrum.shape

        max_radius = self._compute_max_radius(center)
        high_freq_threshold = max_radius * self.HIGH_FREQ_THRESHOLD_RATIO

        #create mask for high frequencies
        outer_mask = create_radial_mask(center, (h, w), max_radius)
        inner_mask = create_radial_mask(center, (h, w), high_freq_threshold)
        high_freq_mask = outer_mask & ~inner_mask

        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask] ** 2)
        total_energy = np.sum(magnitude_spectrum ** 2)

        return high_freq_energy / (total_energy + self.EPSILON)

    def _extract_artifact_features(self,
        magnitude_spectrum: np.ndarray,
        center: Tuple[int, int]
    ) -> np.ndarray:
        """Extract high-frequency artifact features specific to deepfakes.
        Parameters:
            magnitude_spectrum (np.ndarray): magnitude spectrum
            center (Tuple[int, int]): center coordinates
        Returns:
            np.ndarray: artifact features [6]
        """
        h, w = magnitude_spectrum.shape

        #Nyquist frequencies on edges of the spectrum can indicate checkerboard artifacts
        edge = self.ARTIFACT_EDGE_WIDTH
        top_edge = np.mean(magnitude_spectrum[0:edge, :])
        bottom_edge = np.mean(magnitude_spectrum[-edge:, :])
        left_edge = np.mean(magnitude_spectrum[:, 0:edge])
        right_edge = np.mean(magnitude_spectrum[:, -edge:])
        checkerboard_score = (top_edge + bottom_edge + left_edge + right_edge) / 4.0

        #upsampling signatures (periodic peaks in spectrum)
        max_radius = self._compute_max_radius(center)
        num_samples = self.ARTIFACT_NUM_RADIAL_SAMPLES
        radii = np.linspace(0, max_radius, num_samples) 
        radial_profile_samples = []

        for r in radii:
            mask = create_radial_mask(center, (h, w), r)
            values = magnitude_spectrum[mask]
            if len(values) > 0:
                radial_profile_samples.append(np.mean(values))
            else:
                radial_profile_samples.append(0.0)

        upsampling_score = np.var(radial_profile_samples)

        #asymmetry scores (horizontal, vertical)
        left_half = magnitude_spectrum[:, :w//2]
        right_half = magnitude_spectrum[:, w//2:]
        horizontal_asymmetry = np.abs(np.mean(left_half) - np.mean(right_half))

        top_half = magnitude_spectrum[:h//2, :]
        bottom_half = magnitude_spectrum[h//2:, :]
        vertical_asymmetry = np.abs(np.mean(top_half) - np.mean(bottom_half))

        #coherence of medium-high frequency region
        region_size = self.ARTIFACT_CENTER_REGION_SIZE
        high_freq_variance = np.var(magnitude_spectrum[center[0]-region_size:center[0]+region_size,
                                                       center[1]-region_size:center[1]+region_size])

        #spectral irregularity
        smoothed = self._smooth_spectrum(magnitude_spectrum)
        irregularity = np.mean(np.abs(magnitude_spectrum - smoothed))

        features = np.array([
            checkerboard_score,
            upsampling_score,
            horizontal_asymmetry,
            vertical_asymmetry,
            high_freq_variance,
            irregularity
        ], dtype=np.float32)

        return features

    def _smooth_spectrum(self, spectrum: np.ndarray, kernel_size: int = None) -> np.ndarray:
        """Smooth spectrum using simple averaging.
        Parameters:
            spectrum (np.ndarray): input spectrum
            kernel_size (int): smoothing kernel size
        Returns:
            np.ndarray: smoothed spectrum
        """
        if kernel_size is None:
            kernel_size = self.DEFAULT_SMOOTHING_KERNEL_SIZE

        #smoothing box filter
        h, w = spectrum.shape
        pad = kernel_size // 2

        padded = np.pad(spectrum, pad, mode='reflect')

        smoothed = np.zeros_like(spectrum)
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                smoothed[i, j] = np.mean(window)

        return smoothed

    def _compute_cross_band_features(self, radial_features: np.ndarray) -> np.ndarray:
        """Compute cross-band interaction features.
        Parameters:
            radial_features (np.ndarray): radial features array
        Returns:
            np.ndarray: cross-band features [5]
        """
        #band energies from radial features [means, variances, energies, peaks, profile]
        num_bands = self.num_radial_bands
        energies = radial_features[num_bands * self.ENERGY_BAND_START_MULTIPLIER: num_bands * self.ENERGY_BAND_END_MULTIPLIER]

        if len(energies) < 2:
            return np.zeros(5, dtype=np.float32)

        #energy ratios between consecutive bands
        ratio_1_2 = energies[1] / (energies[0] + self.EPSILON)
        ratio_2_3 = energies[2] / (energies[1] + self.EPSILON) if len(energies) > 2 else 0.0
        ratio_low_high = energies[0] / (energies[-1] + self.EPSILON)

        energy_slope = (energies[-1] - energies[0]) / (len(energies) + self.EPSILON)

        #entropy-like measure of energy distribution
        total_energy = np.sum(energies) + self.EPSILON
        energy_dist = energies / total_energy
        energy_concentration = -np.sum(energy_dist * np.log2(energy_dist + self.EPSILON))

        features = np.array([
            ratio_1_2,
            ratio_2_3,
            ratio_low_high,
            energy_slope,
            energy_concentration
        ], dtype=np.float32)

        return features
