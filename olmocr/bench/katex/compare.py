import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def find_image_match(large_pil, small_pil, device=None) -> tuple[float, int, int]:
    """
    Finds the best matching location of a small image inside a large image using 2D convolution.
    Returns a matching score and the coordinates of the best match.

    Args:
        large_pil (PIL.Image): The large image (document).
        small_pil (PIL.Image): The small image (patch).
        device (str, optional): "cuda" for GPU, "cpu" for CPU, or None for auto-selection.

    Returns:
        (score, x, y): 
            - score: Matching score between 0.0 and 1.0, where 1.0 is a perfect match
            - x, y: Coordinates of the top-left corner of the best match location
    """
    
    # Auto-select device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert images to grayscale and NumPy arrays
    large_img = np.array(large_pil.convert("L"), dtype=np.float32) / 255.0
    small_img = np.array(small_pil.convert("L"), dtype=np.float32) / 255.0

    # Swap things around so large image is actually the largest
    if small_img.shape[0] > large_img.shape[0] and small_img.shape[1] > large_img.shape[1]:
        small_img, large_img = large_img, small_img

    # Convert to PyTorch tensors
    large_tensor = torch.tensor(large_img).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    small_tensor = torch.tensor(small_img).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, h, w)

    # Normalize the template (small image) for proper correlation calculation
    # This makes the convolution behave like template matching
    small_mean = torch.mean(small_tensor)
    small_std = torch.std(small_tensor)
    small_normalized = (small_tensor - small_mean) / (small_std + 1e-7)
    
    # Calculate convolution
    def conv2d_fn(large, small):
        return F.conv2d(large, small, padding="same")

    # Perform convolution
    result = conv2d_fn(large_tensor, small_normalized)
    
    # Find the max correlation and its position in a single call
    # result shape is [1, 1, H, W]
    max_val, max_loc = torch.max(result.view(-1), 0)
    
    # Convert flat index to 2D coordinates
    result_size = result.squeeze().size()
    best_y = (max_loc // result_size[1]).item()
    best_x = (max_loc % result_size[1]).item()
    
    # Extract the region from the large image that matches the small image
    h, w = small_img.shape
    
    score = (max_val / torch.mean(large_tensor)).item()
        
    return score, best_x, best_y