import torch
import numpy as np


def denormalize_image_torch(img_tensor):
    """
    Denormalize an image tensor from ImageNet normalization (returns Torch tensor).

    Args:
        img_tensor: Tensor of shape [C, H, W] with ImageNet normalization

    Returns:
        Tensor of shape [C, H, W] with values in [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.detach().cpu() * std + mean
    img = torch.clamp(img, 0, 1)
    return img


def denormalize_image(img_tensor):
    """
    Denormalize an image tensor from ImageNet normalization (returns NumPy array).

    Args:
        img_tensor: Tensor of shape [C, H, W] with ImageNet normalization

    Returns:
        NumPy array of shape [H, W, C] with values in [0, 1]
    """
    img = denormalize_image_torch(img_tensor)
    return img.permute(1, 2, 0).numpy()


def denormalize_image_numpy(img_array):
    """
    Denormalize an image numpy array from ImageNet normalization.

    Args:
        img_array: NumPy array of shape [C, H, W] or [H, W, C] with ImageNet normalization

    Returns:
        NumPy array with values in [0, 1]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Handle both [C, H, W] and [H, W, C] formats
    if img_array.shape[0] == 3:  # [C, H, W] format
        mean = mean.reshape(3, 1, 1)
        std = std.reshape(3, 1, 1)
    else:  # [H, W, C] format
        mean = mean.reshape(1, 1, 3)
        std = std.reshape(1, 1, 3)

    img = img_array * std + mean
    img = np.clip(img, 0, 1)
    return img