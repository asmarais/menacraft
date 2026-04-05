"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation

This module provides a clean implementation of Grad-CAM for visualizing
which regions of an image are important for predictions in CNNs.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import cv2


class GradCAM:
    """
    Grad-CAM implementation for generating class activation maps.

    The algorithm:
    1. Forward pass: Extract feature maps A from target layer [batch, C, H, W]
    2. Backward pass: Compute gradients ∂y/∂A with respect to target class
    3. Global Average Pooling: α_k = (1/Z) Σ_i,j ∂y/∂A_k^(i,j) for each channel k
    4. Weighted combination: L = ReLU(Σ_k α_k · A_k)
    5. Upsample L to input image size

    Attributes:
        model: The neural network model
        target_layer: The convolutional layer to visualize
        activations: Stored feature maps from forward pass
        gradients: Stored gradients from backward pass
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM.

        Args:
            model: The neural network model (must be in eval mode for inference)
            target_layer: The target convolutional layer (typically the last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks to capture activations and gradients
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            """Captures activations during forward pass."""
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            """Captures gradients during backward pass."""
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate class activation map for the input.

        Args:
            input_tensor: Input tensor of shape [batch, channels, height, width]
            target_class: Target class index for gradient computation.
                         If None, uses the predicted class (max logit).

        Returns:
            CAM heatmap of shape [batch, H_cam, W_cam] with values in [0, 1]
            where H_cam, W_cam are the spatial dimensions of the target layer.
        """
        self.model.eval()

        # Ensure gradients are enabled
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        # Handle different output formats
        if isinstance(output, tuple):
            # If model returns (features, pooled_features), use pooled for classification
            logits = output[1]
        else:
            logits = output

        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        batch_size = logits.size(0)
        one_hot = torch.zeros_like(logits)

        if isinstance(target_class, int):
            one_hot[:, target_class] = 1
        else:
            one_hot.scatter_(1, target_class.unsqueeze(1), 1)

        logits.backward(gradient=one_hot, retain_graph=True)

        # Generate CAM
        cam = self._compute_cam()

        return cam

    def _compute_cam(self) -> torch.Tensor:
        """
        Compute the class activation map from stored activations and gradients.

        Returns:
            CAM tensor of shape [batch, H, W] normalized to [0, 1]
        """
        # activations: [batch, channels, H, W]
        # gradients: [batch, channels, H, W]

        # Step 1: Global average pooling of gradients to get weights α_k
        # α_k = (1/Z) Σ_i,j ∂y/∂A_k^(i,j)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [batch, channels, 1, 1]

        # Step 2: Weighted combination of activation maps
        # L = Σ_k α_k · A_k
        cam = torch.sum(weights * self.activations, dim=1)  # [batch, H, W]

        # Step 3: Apply ReLU to focus on positive influence
        cam = F.relu(cam)

        # Step 4: Normalize each sample in batch to [0, 1] - vectorized
        batch_size = cam.size(0)

        # Compute min/max per sample
        cam_flat = cam.view(batch_size, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1)

        # Normalize with safe division
        cam_range = cam_max - cam_min
        mask = cam_range > 1e-8
        cam = torch.where(mask, (cam - cam_min) / cam_range, torch.zeros_like(cam))

        return cam

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        upsample_to: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Generate and optionally upsample CAM.

        Args:
            input_tensor: Input tensor [batch, channels, height, width]
            target_class: Target class for visualization
            upsample_to: Optional target size (H, W) for upsampling

        Returns:
            CAM heatmap, upsampled if upsample_to is specified
        """
        cam = self.generate_cam(input_tensor, target_class)

        if upsample_to is not None:
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=upsample_to,
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

        return cam


class GradCAMVisualizer:
    """
    Utility class for visualizing Grad-CAM heatmaps overlayed on images.
    """

    @staticmethod
    def apply_colormap(
        heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Apply colormap to heatmap.

        Args:
            heatmap: Grayscale heatmap array [H, W] with values in [0, 255]
            colormap: OpenCV colormap (default: COLORMAP_JET)

        Returns:
            Colored heatmap [H, W, 3] in RGB format
        """
        heatmap_uint8 = heatmap.astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        # Convert BGR to RGB
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        return colored_heatmap

    @staticmethod
    def overlay_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            image: Original image [H, W, 3] in RGB, values in [0, 255]
            heatmap: CAM heatmap [H, W] with values in [0, 1]
            alpha: Transparency of heatmap overlay (0=invisible, 1=opaque)

        Returns:
            Overlayed image [H, W, 3] in RGB format
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Resize heatmap to match image size if needed
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Convert heatmap to [0, 255] and apply colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colored_heatmap = GradCAMVisualizer.apply_colormap(heatmap_uint8)

        # Overlay
        overlayed = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)

        return overlayed

    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to NumPy array for visualization.

        Args:
            tensor: Tensor of shape [C, H, W] or [H, W]

        Returns:
            NumPy array in [H, W, C] or [H, W] format
        """
        array = tensor.detach().cpu().numpy()

        if array.ndim == 3:  # [C, H, W] -> [H, W, C]
            array = np.transpose(array, (1, 2, 0))

        return array

    @staticmethod
    def create_visualization(
        original_image: torch.Tensor,
        cam: torch.Tensor,
        alpha: float = 0.4,
        denormalize: bool = True
    ) -> np.ndarray:
        """
        Create complete Grad-CAM visualization.

        Args:
            original_image: Original input tensor [C, H, W] or [H, W, C]
            cam: CAM heatmap tensor [H, W]
            alpha: Overlay transparency
            denormalize: Whether to denormalize image (ImageNet stats)

        Returns:
            Visualization as NumPy array [H, W, 3]
        """
        # Convert tensors to numpy
        if original_image.ndim == 3 and original_image.shape[0] == 3:
            # [C, H, W] format
            image_np = GradCAMVisualizer.tensor_to_numpy(original_image)
        else:
            image_np = original_image.detach().cpu().numpy()

        cam_np = cam.detach().cpu().numpy()

        # Denormalize if needed (ImageNet normalization)
        if denormalize and image_np.ndim == 3:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = std * image_np + mean
            image_np = np.clip(image_np, 0, 1)

        # Ensure image is in [0, 255] uint8 format
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        # Create overlay
        visualization = GradCAMVisualizer.overlay_heatmap(image_np, cam_np, alpha)

        return visualization


class GradCam:
    """Alias for backward compatibility."""
    pass
