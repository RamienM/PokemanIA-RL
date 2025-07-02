import numpy as np

# Mapping of class names to RGB colors
COLOR_DICT = {
    'background': (0, 0, 0),
    'capture': (0, 255, 0),
    'read': (255, 255, 0),
    'main': (255, 0, 0),
    'map': (125, 125, 125),
    'hm': (0, 0, 255),
    'non_walkable': (255, 255, 255),
    'npc': (255, 0, 255),
    'poke_ball': (0, 255, 255),
    'door': (255, 165, 0),
    'heal': (0, 128, 0),
    'ledge': (0, 0, 128),
    'selected': (139, 0, 139),
    'text': (160, 82, 45),
}

# List of class names
CLASS_NAMES = list(COLOR_DICT.keys())

# Normalize color values to the [0, 1] range
COLORS = np.array(list(COLOR_DICT.values())) / 255.0

def apply_overlay(np_image, prediction):
    """
    Apply a colored overlay to the input image based on the predicted segmentation mask.

    Args:
        np_image (np.ndarray): Original input image, values in range [0, 255], shape (H, W) or (H, W, 1).
        prediction (np.ndarray): 2D array with class indices for each pixel, shape (H, W).

    Returns:
        np.ndarray: RGB image with overlay visualization, values in [0, 1].
    """
    # Create an empty RGB mask with the same height and width as the prediction
    mask_rgb = np.zeros((*prediction.shape, 3), dtype=np.float32)

    # Assign colors to the mask based on class ID (skipping class 0: background)
    for class_id, color in enumerate(COLORS):
        if class_id == 0:
            continue
        mask_rgb[prediction == class_id] = color

    # Convert grayscale input image to RGB if needed
    if np_image.ndim == 3 and np_image.shape[-1] == 1:
        np_image = np_image.squeeze(-1)
    img_rgb = np.stack([np_image / 255.0] * 3, axis=-1)

    # Blend the original image and the color mask
    alpha = 0.5
    return (1 - alpha) * img_rgb + alpha * mask_rgb
