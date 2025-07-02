import torch
import numpy as np

def preprocess_image(np_image):
    """
    Preprocess a NumPy image array for model input.

    Converts the image to a float32 PyTorch tensor in [C, H, W] format
    and normalizes pixel values to the [0, 1] range if necessary.

    Args:
        np_image (np.ndarray): Input image. Must be in shape [C, H, W] or [H, W, C] 
                               and have 1 to 4 channels.

    Returns:
        torch.Tensor: Preprocessed image tensor with shape [C, H, W] and float32 values in [0, 1].

    Raises:
        ValueError: If the image shape is not recognized as [C, H, W] or [H, W, C].
    """
    # Handle [C, H, W] format
    if np_image.ndim == 3 and np_image.shape[0] <= 4:
        tensor = torch.from_numpy(np_image).float()
    
    # Handle [H, W, C] format
    elif np_image.ndim == 3 and np_image.shape[2] <= 4:
        tensor = torch.from_numpy(np_image).permute(2, 0, 1).float()
    
    # Invalid format
    else:
        raise ValueError("Invalid image format. Use [C, H, W] or [H, W, C]")

    # Normalize to [0, 1] if necessary
    if tensor.max() > 1:
        tensor = tensor / 255.0

    return tensor
