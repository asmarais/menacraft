import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import logging
import numpy as np
import math
from .base_dct_extractor import BaseDCTExtractor


class TorchDCTExtractor(nn.Module, BaseDCTExtractor):
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
        """Initialize TorchDCTExtractor.
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
        nn.Module.__init__(self)
        BaseDCTExtractor.__init__(
            self,
            block_size=block_size,
            channel_mode=channel_mode,
            aggregation_method=aggregation_method,
            num_frequency_bands=num_frequency_bands,
            feature_dim=feature_dim,
            logger_name=("/".join(__file__.split("/")[-2:])),
            epsilon=epsilon
        )

        #precompute and register DCT transformation matrix as buffer
        dct_matrix = self._create_dct_matrix(block_size)
        self.register_buffer('dct_matrix', dct_matrix)
        """Precomputed DCT transformation matrix"""

        #precompute and register zigzag indices for block traversal
        zigzag_idx = self._create_zigzag_indices(block_size)
        self.register_buffer('zigzag_idx', zigzag_idx)
        """Precomputed zigzag indices for JPEG-style coefficient ordering"""

        self.logger.info("TorchDCTExtractor initialized successfully.")

    def _create_dct_matrix(self, n: int) -> torch.Tensor:
        """Create DCT Type-II transformation matrix.
        Parameters:
            n (int): matrix size (n x n)
        Returns:
            torch.Tensor: DCT transformation matrix [n, n]
        """
        dct_matrix = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    dct_matrix[k, i] = math.sqrt(1.0 / n)
                else:
                    dct_matrix[k, i] = math.sqrt(2.0 / n) * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
        return dct_matrix

    def _create_zigzag_indices(self, n: int) -> torch.Tensor:
        """Create zigzag scan indices for n*n block (JPEG-style DCT coefficient ordering).
        Parameters:
            n (int): block size (n x n)
        Returns:
            torch.Tensor: zigzag indices [n*n, 2] with (row, col) pairs
        """
        indices = []

        #traverse diagonals
        for sum_val in range(2 * n - 1):
            if sum_val % 2 == 0:
                #even diagonals: traverse bottom-left to top-right
                for i in range(min(sum_val, n - 1), max(-1, sum_val - n), -1):
                    j = sum_val - i
                    if 0 <= i < n and 0 <= j < n:
                        indices.append([i, j])
            else:
                #odd diagonals: traverse top-right to bottom-left
                for i in range(max(0, sum_val - n + 1), min(sum_val + 1, n)):
                    j = sum_val - i
                    if 0 <= i < n and 0 <= j < n:
                        indices.append([i, j])

        return torch.tensor(indices, dtype=torch.long)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DCT features from batch of images.
        Parameters:
            images (torch.Tensor): input images [batch, channels, height, width]
                - Expected: RGB images in PyTorch format (channels first)
                - Values in [0, 1] or [-1, 1]
        Returns:
            torch.Tensor: DCT features of shape [batch, feature_dim]
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

        #preprocess images to target channel mode
        preprocessed = self._preprocess_images(images)

        #extract DCT blocks from entire batch
        dct_blocks = self._extract_dct_blocks_batch(preprocessed)

        #aggregate block features
        raw_features = self._aggregate_block_features(dct_blocks)

        #project to target dimension using learnable projection
        projected_features = self.projection(raw_features)

        self.logger.debug(f"- extract - Extracted DCT features with shape {projected_features.shape}")

        return projected_features

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to appropriate channel mode.
        Parameters:
            images (torch.Tensor): input images [batch, 3, height, width]
        Returns:
            torch.Tensor: preprocessed images in target channel format [batch, height, width]
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

            #normalize to [0, 255] range for DCT (matching cv2.dct behavior)
            if channel_images.max() <= 1.0:
                channel_images = channel_images * 255.0

            return channel_images

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

    def _extract_dct_blocks_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Divide images into non-overlapping blocks and apply DCT to each.
        Parameters:
            images (torch.Tensor): preprocessed images [batch, height, width]
        Returns:
            torch.Tensor: DCT coefficient blocks [batch, num_blocks_h, num_blocks_w, block_size, block_size]
        Raises:
            ValueError: if image dimensions are not compatible with block size
        """
        batch_size, h, w = images.shape
        block_size = self.block_size

        #calculate number of blocks (truncate if image size not divisible)
        num_blocks_h = h // block_size
        num_blocks_w = w // block_size

        if num_blocks_h == 0 or num_blocks_w == 0:
            self.logger.error(f"- _extract_dct_blocks_batch - Images too small ({h}x{w}) for block size {block_size}")
            raise ValueError(f"Image dimensions {h}x{w} too small for block size {block_size}")

        try:
            #truncate images to be divisible by block_size
            h_truncated = num_blocks_h * block_size
            w_truncated = num_blocks_w * block_size
            images_truncated = images[:, :h_truncated, :w_truncated]

            #reshape into blocks: [batch, num_blocks_h, block_size, num_blocks_w, block_size]
            blocks = images_truncated.view(batch_size, num_blocks_h, block_size, num_blocks_w, block_size)
            #permute to: [batch, num_blocks_h, num_blocks_w, block_size, block_size]
            blocks = blocks.permute(0, 1, 3, 2, 4).contiguous()

            #flatten batch and block dimensions for DCT computation
            blocks_flat = blocks.view(-1, block_size, block_size)

            #apply 2D DCT to all blocks at once
            dct_blocks = self._apply_dct2d(blocks_flat)

            #reshape back to [batch, num_blocks_h, num_blocks_w, block_size, block_size]
            dct_blocks = dct_blocks.view(batch_size, num_blocks_h, num_blocks_w, block_size, block_size)

        except Exception as e:
            self.logger.error(f"- _extract_dct_blocks_batch - Error applying DCT: {e}")
            raise RuntimeError(f"Error extracting DCT blocks: {e}")

        self.logger.debug(f"- _extract_dct_blocks_batch - Extracted {num_blocks_h * num_blocks_w} DCT blocks per image")
        return dct_blocks

    def _apply_dct2d(self, blocks: torch.Tensor) -> torch.Tensor:
        """Apply 2D DCT to blocks using precomputed transformation matrix.
        Parameters:
            blocks (torch.Tensor): input blocks [num_blocks, block_size, block_size]
        Returns:
            torch.Tensor: DCT coefficients [num_blocks, block_size, block_size]
        """
        #2D DCT: Y = D * X * D^T, where D is the DCT matrix
        dct_matrix = self.dct_matrix
        dct_matrix_t = dct_matrix.t()

        #apply DCT: Y = D @ X @ D^T
        dct_result = torch.matmul(dct_matrix.unsqueeze(0), blocks)
        dct_result = torch.matmul(dct_result, dct_matrix_t.unsqueeze(0))

        return dct_result

    def _aggregate_block_features(self, dct_blocks: torch.Tensor) -> torch.Tensor:
        """Aggregate DCT blocks into fixed-size feature vector.
        Parameters:
            dct_blocks (torch.Tensor): DCT coefficient blocks [batch, num_blocks_h, num_blocks_w, block_size, block_size]
        Returns:
            torch.Tensor: aggregated feature vector [batch, num_raw_features]
        Raises:
            ValueError: if dct_blocks is empty
        """
        try:
            assert dct_blocks.numel() > 0, "No DCT blocks to aggregate"

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

    def _frequency_bands_aggregation(self, dct_blocks: torch.Tensor) -> torch.Tensor:
        """Aggregate DCT blocks by frequency bands.
        Parameters:
            dct_blocks (torch.Tensor): DCT coefficient blocks [batch, num_blocks_h, num_blocks_w, block_size, block_size]
        Returns:
            torch.Tensor: aggregated features [batch, num_raw_features]
        """
        batch_size, num_blocks_h, num_blocks_w, block_size, _ = dct_blocks.shape
        num_bands = self.num_frequency_bands
        device = dct_blocks.device

        #flatten blocks: [batch, num_blocks, block_size, block_size]
        blocks_flat = dct_blocks.view(batch_size, -1, block_size, block_size)
        num_blocks = blocks_flat.shape[1]

        #initialize band statistics
        band_means = torch.zeros(batch_size, num_bands, device=device)
        band_stds = torch.zeros(batch_size, num_bands, device=device)
        band_energies = torch.zeros(batch_size, num_bands, device=device)

        #create frequency band masks based on radial distance from DC
        max_distance = math.sqrt(2) * (block_size - 1)
        band_width = max_distance / num_bands

        #compute distance map for a single block (all blocks have same structure)
        y = torch.arange(block_size, device=device).view(-1, 1).float()
        x = torch.arange(block_size, device=device).view(1, -1).float()
        distance_map = torch.sqrt(x**2 + y**2)

        #collect coefficients for each band across all blocks
        for band_idx in range(num_bands):
            #define band range
            min_dist = band_idx * band_width
            max_dist = (band_idx + 1) * band_width

            #extract coefficients in this band
            mask = (distance_map >= min_dist) & (distance_map < max_dist)

            #extract coefficients from all blocks in batch
            band_coeffs = blocks_flat[:, :, mask]  # [batch, num_blocks, num_coeffs_in_band]

            #flatten to [batch, num_blocks * num_coeffs_in_band]
            band_coeffs_flat = band_coeffs.reshape(batch_size, -1)

            if band_coeffs_flat.shape[1] > 0:
                band_means[:, band_idx] = torch.mean(band_coeffs_flat, dim=1)
                band_stds[:, band_idx] = torch.std(band_coeffs_flat, dim=1)
                band_energies[:, band_idx] = torch.sum(band_coeffs_flat ** 2, dim=1)

        #global statistics
        all_dc_coeffs = blocks_flat[:, :, 0, 0]  # [batch, num_blocks]
        all_ac_coeffs = blocks_flat[:, :, 1:, :].reshape(batch_size, num_blocks, -1)  # [batch, num_blocks, ac_size]

        mean_dc = torch.mean(all_dc_coeffs, dim=1)
        std_dc = torch.std(all_dc_coeffs, dim=1)
        mean_ac = torch.mean(all_ac_coeffs, dim=[1, 2])
        std_ac = torch.std(all_ac_coeffs.reshape(batch_size, -1), dim=1)

        #high frequency ratio (energy in highest band / total energy)
        total_energy = torch.sum(band_energies, dim=1)
        high_freq_ratio = band_energies[:, -1] / (total_energy + self.EPSILON)

        #histogram of coefficient magnitudes (10 bins) - vectorized
        all_coeffs = blocks_flat.reshape(batch_size, -1)
        all_coeffs_abs = torch.abs(all_coeffs)  # [batch, num_coeffs]
        bins = 10
        hist_min, hist_max = 0.0, 100.0
        bin_width = (hist_max - hist_min) / bins

        # Compute bin indices [batch, num_coeffs]
        all_coeffs_clamped = torch.clamp(all_coeffs_abs, hist_min, hist_max - 1e-6)
        bin_indices = ((all_coeffs_clamped - hist_min) / bin_width).long()
        bin_indices = torch.clamp(bin_indices, 0, bins - 1)

        # One-hot encoding [batch, num_coeffs, bins]
        one_hot = F.one_hot(bin_indices, num_classes=bins).float()

        # Sum to get histogram [batch, bins]
        hist = one_hot.sum(dim=1)

        # Normalize
        hist = hist / (hist.sum(dim=1, keepdim=True) + self.EPSILON)

        #concatenate all features
        features = torch.cat([
            band_means,      # num_bands features
            band_stds,       # num_bands features
            band_energies,   # num_bands features
            mean_dc.unsqueeze(1),
            std_dc.unsqueeze(1),
            mean_ac.unsqueeze(1),
            std_ac.unsqueeze(1),
            high_freq_ratio.unsqueeze(1),  # 5 global features
            hist             # 10 histogram features
        ], dim=1)

        return features

    def _zigzag_aggregation(self, dct_blocks: torch.Tensor) -> torch.Tensor:
        """Aggregate DCT blocks using zigzag ordering (JPEG-style).
        Parameters:
            dct_blocks (torch.Tensor): DCT coefficient blocks [batch, num_blocks_h, num_blocks_w, block_size, block_size]
        Returns:
            torch.Tensor: aggregated features [batch, num_raw_features]
        """
        batch_size, num_blocks_h, num_blocks_w, block_size, _ = dct_blocks.shape
        device = dct_blocks.device

        #flatten blocks: [batch, num_blocks, block_size, block_size]
        blocks_flat = dct_blocks.view(batch_size, -1, block_size, block_size)
        num_blocks = blocks_flat.shape[1]

        #extract top 16 coefficients in zigzag order from each block
        zigzag_idx = self.zigzag_idx[:16]  # [16, 2]

        #gather coefficients using zigzag indices
        zigzag_coeffs = blocks_flat[:, :, zigzag_idx[:, 0], zigzag_idx[:, 1]]  # [batch, num_blocks, 16]

        #compute statistics across blocks for each coefficient position
        means = torch.mean(zigzag_coeffs, dim=1)  # [batch, 16]

        #histogram of coefficient magnitudes - vectorized
        all_coeffs = zigzag_coeffs.reshape(batch_size, -1)
        all_coeffs_abs = torch.abs(all_coeffs)  # [batch, num_coeffs]
        bins = 10
        hist_min, hist_max = 0.0, 100.0
        bin_width = (hist_max - hist_min) / bins

        # Compute bin indices [batch, num_coeffs]
        all_coeffs_clamped = torch.clamp(all_coeffs_abs, hist_min, hist_max - 1e-6)
        bin_indices = ((all_coeffs_clamped - hist_min) / bin_width).long()
        bin_indices = torch.clamp(bin_indices, 0, bins - 1)

        # One-hot encoding [batch, num_coeffs, bins]
        one_hot = F.one_hot(bin_indices, num_classes=bins).float()

        # Sum to get histogram [batch, bins]
        hist = one_hot.sum(dim=1)

        # Normalize
        hist = hist / (hist.sum(dim=1, keepdim=True) + self.EPSILON)

        features = torch.cat([means, hist], dim=1)
        return features

    def _statistical_aggregation(self, dct_blocks: torch.Tensor) -> torch.Tensor:
        """Statistical aggregation across spatial block positions.
        Parameters:
            dct_blocks (torch.Tensor): DCT coefficient blocks [batch, num_blocks_h, num_blocks_w, block_size, block_size]
        Returns:
            torch.Tensor: aggregated features [batch, num_raw_features]
        """
        batch_size, num_blocks_h, num_blocks_w, block_size, _ = dct_blocks.shape

        #flatten blocks: [batch, num_blocks, block_size, block_size]
        blocks_flat = dct_blocks.view(batch_size, -1, block_size, block_size)

        #compute statistics across blocks for each coefficient position
        means = torch.mean(blocks_flat, dim=1).flatten(1)  # [batch, block_size^2]
        stds = torch.std(blocks_flat, dim=1).flatten(1)    # [batch, block_size^2]
        energies = torch.sum(blocks_flat ** 2, dim=1).flatten(1)  # [batch, block_size^2]

        features = torch.cat([means, stds, energies], dim=1)
        return features
