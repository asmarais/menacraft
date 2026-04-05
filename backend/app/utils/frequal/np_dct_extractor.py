import cv2
import numpy as np
import torch
from typing import List
import logging
from .utils import convert_to_luminance, convert_to_grayscale, zigzag_indices, format_image
from .base_dct_extractor import BaseDCTExtractor


class NpDCTExtractor(BaseDCTExtractor):
    """Extract DCT-based frequency features from face images to detect JPEG compression
    artifacts and frequency domain anomalies commonly found in manipulated faces."""

    def __init__(self,
        block_size: int,
        channel_mode: str,
        aggregation_method: str,
        num_frequency_bands: int,
        feature_dim: int,
        epsilon: float = 1e-10
    ):
        """Initialize NpDCTExtractor.
        Parameters:
            block_size (int): size of DCT blocks in pixels
            channel_mode (str): how to handle color channels
            aggregation_method (str): how to aggregate block statistics
            num_frequency_bands (int): number of frequency bands to extract
            feature_dim (int): target output feature dimension
            epsilon (float): small constant for numerical stability
        Raises:
            ValueError: if parameters are invalid
        """
        super().__init__(
            block_size=block_size,
            channel_mode=channel_mode,
            aggregation_method=aggregation_method,
            num_frequency_bands=num_frequency_bands,
            feature_dim=feature_dim,
            logger_name=("/".join(__file__.split("/")[-2:])),
            epsilon=epsilon
        )

        #generate zigzag indices for block traversal (cached for efficiency)
        self._zigzag_idx = zigzag_indices(block_size)

        self.logger.info("DCTExtractor initialized successfully.")

    def extract(self, images: np.ndarray) -> np.ndarray:
        """Extract DCT frequency features from batch of images.
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
                dct_blocks = self._extract_dct_blocks(preprocessed)
                raw_features = self._aggregate_block_features(dct_blocks)

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
        """Convert image to appropriate channel mode.
        Parameters:
            image (np.ndarray): input image [height, width, channels] or [height, width]
        Returns:
            np.ndarray: preprocessed image in target channel format [height, width]
        Raises:
            ValueError: if image format is invalid
        """
        try:
            image = format_image(image.copy())

            #convert to target channel mode
            match self.channel_mode:
                case "luminance":
                    return convert_to_luminance(image)
                case "grayscale":
                    return convert_to_grayscale(image)
                case "per_channel":
                    #per_channel mode not fully implemented
                    self.logger.warning("- _preprocess_image - per_channel mode not fully supported, using luminance")
                    return convert_to_luminance(image)
                case _:
                    raise ValueError(f"Unknown channel mode: {self.channel_mode}")

        except Exception as e:
            self.logger.error(f"- _preprocess_image - {e}")
            raise ValueError(f"Error preprocessing image: {e}")

    def _extract_dct_blocks(self, image: np.ndarray) -> List[np.ndarray]:
        """Divide image into non-overlapping blocks and apply DCT to each.
        Parameters:
            image (np.ndarray): preprocessed image [height, width]
        Returns:
            List[np.ndarray]: list of DCT coefficient blocks, each of shape [block_size, block_size]
        Raises:
            ValueError: if image dimensions are not compatible with block size
        """
        h, w = image.shape
        block_size = self.block_size

        #calculate number of blocks (truncate if image size not divisible)
        num_blocks_h = h // block_size
        num_blocks_w = w // block_size

        if num_blocks_h == 0 or num_blocks_w == 0:
            self.logger.error(f"- _extract_dct_blocks - Image too small ({h}x{w}) for block size {block_size}")
            raise ValueError(f"Image dimensions {h}x{w} too small for block size {block_size}")

        dct_blocks = []

        try:
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    #extract block
                    y_start = i * block_size
                    x_start = j * block_size
                    block = image[y_start:y_start + block_size, x_start:x_start + block_size]

                    #convert to float32 for DCT
                    block_float = block.astype(np.float32)

                    #apply 2D DCT
                    dct_block = cv2.dct(block_float)
                    dct_blocks.append(dct_block)

        except Exception as e:
            self.logger.error(f"- _extract_dct_blocks - Error applying DCT: {e}")
            raise RuntimeError(f"Error extracting DCT blocks: {e}")

        self.logger.debug(f"- _extract_dct_blocks - Extracted {len(dct_blocks)} DCT blocks")
        return dct_blocks

    def _aggregate_block_features(self, dct_blocks: List[np.ndarray]) -> np.ndarray:
        """Aggregate DCT blocks into fixed-size feature vector.
        Parameters:
            dct_blocks (List[np.ndarray]): list of DCT coefficient blocks
        Returns:
            np.ndarray: aggregated feature vector [num_raw_features]
        Raises:
            ValueError: if dct_blocks is empty
        """
        try:
            assert len(dct_blocks) > 0, "No DCT blocks to aggregate"

        except AssertionError as e:
            self.logger.error(f"- _aggregate_block_features - {e}")
            raise ValueError(f"Invalid DCT blocks: {e}")

        match self.aggregation_method:
            case "frequency_bands":
                return self._frequency_bands_aggregation(dct_blocks)
            case "zigzag":
                return self._zigzag_aggregation(dct_blocks)
            case _:  #statistical
                return self._statistical_aggregation(dct_blocks)

    def _frequency_bands_aggregation(self, dct_blocks: List[np.ndarray]) -> np.ndarray:
        """Aggregate DCT blocks by frequency bands.
        Parameters:
            dct_blocks (List[np.ndarray]): list of DCT coefficient blocks
        Returns:
            np.ndarray: aggregated features [num_raw_features]
        """
        block_size = self.block_size
        num_bands = self.num_frequency_bands

        #initialize band statistics
        band_means = np.zeros(num_bands)
        band_stds = np.zeros(num_bands)
        band_energies = np.zeros(num_bands)

        #create frequency band masks based on radial distance from DC
        max_distance = np.sqrt(2) * (block_size - 1)
        band_width = max_distance / num_bands

        #compute distance map for a single block (all blocks have same structure)
        y, x = np.ogrid[:block_size, :block_size]
        distance_map = np.sqrt(x**2 + y**2)

        #collect coefficients for each band across all blocks
        band_coefficients = [[] for _ in range(num_bands)]

        for block in dct_blocks:
            for band_idx in range(num_bands):
                #define band range
                min_dist = band_idx * band_width
                max_dist = (band_idx + 1) * band_width

                #extract coefficients in this band
                mask = (distance_map >= min_dist) & (distance_map < max_dist)
                coeffs = block[mask]
                band_coefficients[band_idx].extend(coeffs)

        #compute statistics for each band
        for band_idx in range(num_bands):
            coeffs = np.array(band_coefficients[band_idx])
            if len(coeffs) > 0:
                band_means[band_idx] = np.mean(coeffs)
                band_stds[band_idx] = np.std(coeffs)
                band_energies[band_idx] = np.sum(coeffs ** 2)

        #global statistics
        all_dct_coeffs = np.array([block.flatten() for block in dct_blocks]).flatten()
        all_dc_coeffs = np.array([block[0, 0] for block in dct_blocks])
        all_ac_coeffs = np.array([block[1:, :].flatten() for block in dct_blocks]).flatten()

        mean_dc = np.mean(all_dc_coeffs)
        std_dc = np.std(all_dc_coeffs)
        mean_ac = np.mean(all_ac_coeffs)
        std_ac = np.std(all_ac_coeffs)

        #high frequency ratio (energy in highest band / total energy)
        total_energy = np.sum(band_energies)
        high_freq_ratio = band_energies[-1] / (total_energy + self.EPSILON)

        #histogram of coefficient magnitudes (10 bins)
        hist, _ = np.histogram(np.abs(all_dct_coeffs), bins=10, range=(0, 100), density=True)

        #concatenate all features
        features = np.concatenate([
            band_means,      # num_bands features
            band_stds,       # num_bands features
            band_energies,   # num_bands features
            [mean_dc, std_dc, mean_ac, std_ac, high_freq_ratio],  # 5 global features
            hist             # 10 histogram features
        ])

        return features.astype(np.float32)

    def _zigzag_aggregation(self, dct_blocks: List[np.ndarray]) -> np.ndarray:
        """Aggregate DCT blocks using zigzag ordering (JPEG-style).
        Parameters:
            dct_blocks (List[np.ndarray]): list of DCT coefficient blocks
        Returns:
            np.ndarray: aggregated features [num_raw_features]
        """
        #extract top 16 coefficients in zigzag order from each block
        zigzag_coeffs = []

        for block in dct_blocks:
            #get coefficients in zigzag order
            coeffs_zigzag = [block[i, j] for i, j in self._zigzag_idx[:16]]
            zigzag_coeffs.append(coeffs_zigzag)

        zigzag_coeffs = np.array(zigzag_coeffs)

        #compute statistics across blocks for each coefficient position
        means = np.mean(zigzag_coeffs, axis=0)

        #histogram of coefficient magnitudes
        all_coeffs = zigzag_coeffs.flatten()
        hist, _ = np.histogram(np.abs(all_coeffs), bins=10, range=(0, 100), density=True)

        features = np.concatenate([means, hist])
        return features.astype(np.float32)

    def _statistical_aggregation(self, dct_blocks: List[np.ndarray]) -> np.ndarray:
        """Statistical aggregation across spatial block positions.
        Parameters:
            dct_blocks (List[np.ndarray]): list of DCT coefficient blocks
        Returns:
            np.ndarray: aggregated features [num_raw_features]
        """
        block_size = self.block_size

        #stack all blocks into 3D array [num_blocks, block_size, block_size]
        blocks_array = np.array(dct_blocks)

        #compute statistics across blocks for each coefficient position
        means = np.mean(blocks_array, axis=0).flatten()
        stds = np.std(blocks_array, axis=0).flatten()
        energies = np.sum(blocks_array ** 2, axis=0).flatten()

        features = np.concatenate([means, stds, energies])
        return features.astype(np.float32)
