import unittest
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def find_image_match(large_pil, small_pil, device=None) -> tuple[float, int, int]:
    """
    Finds the best matching location of a small image inside a large image using 2D convolution.
    Returns a matching score and the coordinates of the best match.
    
    The matching score is computed using Intersection over Union (IoU) of binary images.
    Each image is converted to a binary mask where pixels > 0.5 are True, otherwise False.
    The IoU is calculated as: (intersection) / (union) of the two binary masks.
    
    If the extracted patch and the template differ in shape (which can happen when the patch goes out of bounds),
    the smaller array is padded with False values so that they can be compared elementwise.
    
    Args:
        large_pil (PIL.Image): The large image.
        small_pil (PIL.Image): The small image (patch).
        device (str, optional): "cuda" or "cpu". If None, auto-select based on availability.
    
    Returns:
        (score, x, y): 
            - score: IoU matching score (0.0 to 1.0)
            - x, y: Coordinates (top-left corner) of the best match in the large image.
    """
    # Auto-select device.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert images to grayscale and normalize to [0, 1]
    large_img = np.array(large_pil.convert("L"), dtype=np.float32) / 255.0
    small_img = np.array(small_pil.convert("L"), dtype=np.float32) / 255.0

    # If the "small" image is larger than the "large" image in any dimension, swap them.
    if small_img.shape[0] > large_img.shape[0] or small_img.shape[1] > large_img.shape[1]:
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

    # If there is a shape mismatch (e.g. when near image boundaries), pad the smaller array with False (0.0)
    if large_patch.shape != small_img.shape:
        target_shape = (max(large_patch.shape[0], small_img.shape[0]),
                        max(large_patch.shape[1], small_img.shape[1]))
        def pad_to_shape(arr, target_shape, pad_value=0.0):
            pad_h = target_shape[0] - arr.shape[0]
            pad_w = target_shape[1] - arr.shape[1]
            return np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=pad_value)
        if large_patch.shape != target_shape:
            large_patch = pad_to_shape(large_patch, target_shape, pad_value=0.0)
        if small_img.shape != target_shape:
            small_img = pad_to_shape(small_img, target_shape, pad_value=0.0)

    # Create binary masks (True if > 0.5, else False)
    large_binary = large_patch > 0.5
    small_binary = small_img > 0.5
    
    # Create masks for very bright pixels (> 0.99)
    large_white = large_patch > 0.99
    small_white = small_img > 0.99
    
    # Create a mask for pixels to exclude (where both images are very bright)
    exclude_mask = np.logical_and(large_white, small_white)
    
    # Apply the exclusion mask to the binary masks
    large_binary_filtered = np.logical_and(large_binary, ~exclude_mask)
    small_binary_filtered = np.logical_and(small_binary, ~exclude_mask)
    
    # Calculate intersection and union on the filtered binary masks
    intersection = np.logical_and(large_binary_filtered, small_binary_filtered).sum()
    union = np.logical_or(large_binary_filtered, small_binary_filtered).sum()
    
    # Calculate IoU score
    # Handle the case where union is zero (both images empty)
    if union == 0:
        score = 1.0 if intersection == 0 else 0.0
    else:
        score = float(intersection / union)

    return score, match_x, match_y


class TestFindImageMatch(unittest.TestCase):
    def setUp(self):
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
        np.random.seed(123)
        large_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        large_pil = Image.fromarray(large_array, mode='L')
        
        top, left = 30, 40  # expected best_y, best_x
        patch_height, patch_width = 20, 20
        patch_array = large_array[top:top+patch_height, left:left+patch_width]
        small_pil = Image.fromarray(patch_array, mode='L')
        
        score, best_x, best_y = find_image_match(large_pil, small_pil)
        self.assertEqual(best_x, left, f"Expected best_x to be {left} but got {best_x}")
        self.assertEqual(best_y, top, f"Expected best_y to be {top} but got {best_y}")
        self.assertGreater(score, 0.99, f"Expected high score for an exact match, got {score}")

    def test_full_image_match(self):
        """Test when the small image is identical to the large image."""
        large_pil = self.create_random_image((50, 50))
        small_pil = large_pil.copy()
        score, best_x, best_y = find_image_match(large_pil, small_pil)
        self.assertEqual(best_x, 0, f"Expected best_x to be 0, got {best_x}")
        self.assertEqual(best_y, 0, f"Expected best_y to be 0, got {best_y}")
        self.assertGreater(score, 0.99, f"Expected high score for an exact match, got {score}")

    def test_swap_images(self):
        """
        Test the swapping logic by passing in images in reversed order.
        When the "small" image is actually larger than the "large" image,
        the function should swap them internally.
        """
        large_img = self.create_random_image((100, 100))
        large_array = np.array(large_img)
        top, left = 20, 20
        patch_height, patch_width = 40, 40
        patch_array = large_array[top:top+patch_height, left:left+patch_width]
        small_img = Image.fromarray(patch_array, mode='L')
        # Pass in swapped: the larger image as the patch and vice versa.
        score, best_x, best_y = find_image_match(small_img, large_img)
        self.assertEqual(best_x, left, f"Expected best_x to be {left} after swap, got {best_x}")
        self.assertEqual(best_y, top, f"Expected best_y to be {top} after swap, got {best_y}")

    def test_single_pixel_match(self):
        """Test the function with 1x1 images."""
        arr = np.array([[128]], dtype=np.uint8)
        pil_img = Image.fromarray(arr, mode='L')
        score, best_x, best_y = find_image_match(pil_img, pil_img)
        self.assertEqual(best_x, 0)
        self.assertEqual(best_y, 0)
        self.assertGreater(score, 0.99, f"Expected high score for an exact match, got {score}")

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

    def test_padding_mismatch(self):
        """
        Test a case where the computed patch from the large image is smaller than the
        template due to clamping at the boundary. In such cases, the smaller array should be
        padded with white pixels (1.0) so that both arrays have the same shape.
        
        Here, we force a mismatch by providing images whose sizes (after potential swap) cause
        the extracted patch to be truncated.
        """
        # Create a "large" image that is too small in height compared to the template.
        # For example, after swap, effective large image will be (50, 150) and template is (100, 100)
        large_pil = self.create_random_image((50, 150))
        small_pil = self.create_random_image((100, 100))
        # Calling with these images: since small_pil is larger in height than large_pil,
        # a swap will occur. After swap:
        #   effective large image: (100, 100)
        #   effective small image: (50, 150)
        # Then patch size is taken from effective small image: (50, 150)
        # However, the effective large image is (100, 100), so when extracting a patch of size (50,150)
        # from a  (100, 100) image (clamped to width 100), the patch will be (50, 100).
        # Our padding logic should pad the extracted patch from width 100 to 150.
        score, best_x, best_y = find_image_match(large_pil, small_pil)
        # After padding, the per-pixel score should be computed without error.
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
       

class TestRenderMathMatches(unittest.TestCase):
    # def testBasicMatch1(self):
    #     from .render import render_equation

    #     ref_image = render_equation("\int_{a}^{b} f(x) \, dx = F(b) - F(a)")
    #     hyp_image = render_equation("\int_a^b f(x) \, dx = F(b) - F(a)")

    #     score, best_x, best_y = find_image_match(ref_image, hyp_image)
    #     self.assertGreater(score, 0.99)

    # def testBasicMatch2(self):
    #     from .render import render_equation

    #     ref_image = render_equation("s(t) = t^2 + 8t - 1")
    #     hyp_image = render_equation("s(t) = t^2 + 8t + 1")

    #     score, best_x, best_y = find_image_match(ref_image, hyp_image)
    #     print("Should be high diff")
    #     print(score, best_x, best_y)  
    #     self.assertLess(score, 0.95)

    # def testBasicMatch3(self):
    #     from .render import render_equation

    #     ref_image = render_equation("s(t) = t^2 + 8t - 1")

    #     new_image = Image.new(ref_image.mode, (ref_image.width + 20, ref_image.height), (255, 255, 255))
    
    #     # Paste the original image onto the new image, offset by the padding amount
    #     new_image.paste(ref_image, (20, 0))

    #     score, best_x, best_y = find_image_match(ref_image, new_image)
    #     print("Should be exactly the same, shfited over")
    #     print(score, best_x, best_y)        
    #     self.assertGreater(score, 0.99)
    #     self.assertEqual(best_x, 20)
    #     self.assertEqual(best_y, 0)

    # def testBasicMatch4(self):
    #     from .render import render_equation

    #     ref_image = render_equation("s(t) = t^2 + 8t - 1")
    #     hyp_image = render_equation("e^{i\pi} + 1 = 0")

    #     score, best_x, best_y = find_image_match(ref_image, hyp_image)
    #     print("Should be way off")
    #     print(score, best_x, best_y)
    #     self.assertLess(score, 0.5)

    def testMultiline(self):
        from .render import render_equation

        ref_image = render_equation("\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\varepsilon_0}")
        hyp_image = render_equation("""\\begin{align*}\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\varepsilon_0}\\end{align*}""")

        ref_image.save("ref1.png")
        hyp_image.save("hyp1.png")

        score, best_x, best_y = find_image_match(ref_image, hyp_image)
        print("Should be way in there")
        print(score, best_x, best_y)
        self.assertGreater(score, 0.95)



if __name__ == "__main__":
    unittest.main()
