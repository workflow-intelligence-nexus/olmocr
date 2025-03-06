import unittest
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def find_image_match(large_pil, small_pil, device=None) -> tuple[float, int, int]:
    """
    Finds the best matching location of a small image inside a large image using 2D convolution.
    Returns a matching score and the coordinates of the best match.
    
    The matching score is computed by comparing the pixel values in the extracted patch from the large image 
    with the small image. For each pixel, if both values are equal (or both zero) the match is perfect (1.0),
    otherwise the match is the ratio: min(a, b)/max(a, b). The final score is the average over all pixels.
    
    Args:
        large_pil (PIL.Image): The large image.
        small_pil (PIL.Image): The small image (patch).
        device (str, optional): "cuda" or "cpu". If None, auto-select based on availability.
    
    Returns:
        (score, x, y): 
            - score: Average matching score (0.0 to 1.0)
            - x, y: Coordinates (top-left corner) of the best match in the large image.
    """
    # Auto-select device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert images to grayscale and normalize to [0, 1]
    large_img = np.array(large_pil.convert("L"), dtype=np.float32) / 255.0
    small_img = np.array(small_pil.convert("L"), dtype=np.float32) / 255.0

    # If the "small" image is actually larger than the "large" image, swap them.
    if small_img.shape[0] > large_img.shape[0] and small_img.shape[1] > large_img.shape[1]:
        small_img, large_img = large_img, small_img

    # Convert images to torch tensors with shape (1, 1, H, W)
    large_tensor = torch.tensor(large_img).unsqueeze(0).unsqueeze(0).to(device)
    small_tensor = torch.tensor(small_img).unsqueeze(0).unsqueeze(0).to(device)

    # Normalize the small image (template) for proper correlation calculation.
    small_mean = torch.mean(small_tensor)
    small_std = torch.std(small_tensor)
    small_normalized = (small_tensor - small_mean) / (small_std + 1e-7)

    # Perform convolution with same padding.
    result = F.conv2d(large_tensor, small_normalized, padding="same")

    # Find the maximum correlation value and its flat index.
    max_val, max_loc = torch.max(result.view(-1), 0)

    # Extract the coordinates from the convolution result.
    # Handle the scalar case (e.g. for 1x1 images).
    if result.squeeze().dim() == 0:
        conv_y, conv_x = 0, 0
    else:
        result_size = result.squeeze().size()  # expected shape: (H, W)
        conv_y = (max_loc // result_size[1]).item()
        conv_x = (max_loc % result_size[1]).item()

    # Compute the offset introduced by "same" padding.
    patch_h, patch_w = small_img.shape
    offset_y = (patch_h - 1) // 2
    offset_x = (patch_w - 1) // 2

    # Adjust the convolution coordinate to get the top-left corner of the patch.
    match_y = conv_y - offset_y
    match_x = conv_x - offset_x

    # Clamp coordinates to be within valid bounds.
    match_y = max(0, min(match_y, large_img.shape[0] - patch_h))
    match_x = max(0, min(match_x, large_img.shape[1] - patch_w))

    # Extract the corresponding patch from the large image.
    large_patch = large_img[match_y:match_y+patch_h, match_x:match_x+patch_w]

    # Compute per-pixel matching score:
    # If both pixels are zero, the match is perfect (1.0). Otherwise, score = min(a,b) / max(a,b)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where((large_patch == 0) & (small_img == 0), 1.0,
                         np.where((large_patch == 0) | (small_img == 0), 0.0,
                                  np.minimum(large_patch, small_img) / np.maximum(large_patch, small_img)))
    score = float(np.mean(ratio))

    return score, match_x, match_y


class TestFindImageMatch(unittest.TestCase):
    def setUp(self):
        # Fix random seeds for reproducibility.
        np.random.seed(42)
        torch.manual_seed(42)

    def create_random_image(self, shape):
        """
        Create a random grayscale image with the given shape (height, width).
        Pixel values are in the range 0-255.
        """
        arr = np.random.randint(0, 256, shape, dtype=np.uint8)
        return Image.fromarray(arr, mode='L')

    def test_exact_match(self):
        """Test that a patch cropped from a larger image is found at the correct location."""
        # Create a synthetic large image with reproducible random data.
        np.random.seed(123)
        large_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        large_pil = Image.fromarray(large_array, mode='L')
        
        # Crop a patch from a known location.
        top, left = 30, 40  # (row, column) -> expected best_y, best_x
        patch_height, patch_width = 20, 20
        patch_array = large_array[top:top+patch_height, left:left+patch_width]
        small_pil = Image.fromarray(patch_array, mode='L')
        
        score, best_x, best_y = find_image_match(large_pil, small_pil)
        
        self.assertEqual(best_x, left, f"Expected best_x to be {left} but got {best_x}")
        self.assertEqual(best_y, top, f"Expected best_y to be {top} but got {best_y}")
        # For an exact match we expect a relatively high score.
        self.assertGreater(score, 0.9, f"Expected high score for an exact match, got {score}")

    def test_full_image_match(self):
        """Test when the small image is identical to the large image."""
        large_pil = self.create_random_image((50, 50))
        # Use a copy so that both images are the same.
        small_pil = large_pil.copy()
        score, best_x, best_y = find_image_match(large_pil, small_pil)
        
        self.assertEqual(best_x, 0, f"Expected best_x to be 0, got {best_x}")
        self.assertEqual(best_y, 0, f"Expected best_y to be 0, got {best_y}")
        self.assertIsInstance(score, float)

    def test_swap_images(self):
        """
        Test the swapping logic by passing in images in reversed order.
        When the "small" image is actually larger than the "large" image,
        the function should swap them internally.
        """
        # Create a larger image.
        large_img = self.create_random_image((100, 100))
        large_array = np.array(large_img)
        # Crop a patch from the larger image.
        top, left = 20, 20
        patch_height, patch_width = 40, 40
        patch_array = large_array[top:top+patch_height, left:left+patch_width]
        small_img = Image.fromarray(patch_array, mode='L')
        
        # Pass in swapped: the larger image as the patch and vice versa.
        score, best_x, best_y = find_image_match(small_img, large_img)
        
        # After swapping, we expect the match to be found at (left, top)
        self.assertEqual(best_x, left, f"Expected best_x to be {left} after swap, got {best_x}")
        self.assertEqual(best_y, top, f"Expected best_y to be {top} after swap, got {best_y}")

    def test_single_pixel_match(self):
        """Test the function with 1x1 images."""
        # Create a 1x1 image with a mid-gray value.
        arr = np.array([[128]], dtype=np.uint8)
        pil_img = Image.fromarray(arr, mode='L')
        score, best_x, best_y = find_image_match(pil_img, pil_img)
        
        # For a 1x1 image, the best match is at (0,0).
        self.assertEqual(best_x, 0)
        self.assertEqual(best_y, 0)
        self.assertIsInstance(score, float)

    def test_out_of_bounds_coordinates(self):
        """
        Test that the returned best match coordinates are within
        the bounds of the large image.
        """
        large_pil = self.create_random_image((80, 80))
        large_array = np.array(large_pil)
        left, top = 30, 30
        patch_width, patch_height = 20, 20
        patch_array = large_array[top:top+patch_height, left:left+patch_width]
        small_pil = Image.fromarray(patch_array, mode='L')
        score, best_x, best_y = find_image_match(large_pil, small_pil)
        
        width, height = large_pil.size
        self.assertTrue(0 <= best_x < width, f"best_x {best_x} is out of bounds for width {width}")
        self.assertTrue(0 <= best_y < height, f"best_y {best_y} is out of bounds for height {height}")

if __name__ == "__main__":
    unittest.main()
