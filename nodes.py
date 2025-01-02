# nodes.py

from . import find_watermarks
import torchvision.transforms as transforms

# Define transforms for converting between PIL Images and Tensors
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# ------------------------------------------------------------------------------
# Node 1: FindWatermarkNode
# ------------------------------------------------------------------------------
class FindWatermarkNode:
    """
    This node takes a batch of images and returns:
      - The x and y coordinates of the detected watermark
      - The watermark detection score as a string
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),  # Changed from "TENSOR" to "IMAGE"
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("x", "y", "score")
    FUNCTION = "find_watermark"
    CATEGORY = "Video/Watermarks"

    def find_watermark(self, images):
        """
        Args:
            images (list of PIL.Image): Batch of input images.

        Returns:
            Tuple[int, int, str]: x-coordinate, y-coordinate, and detection score.
        """
        # Call your existing watermark detection function
        result = find_watermarks.find_watermark_tensor(images.permute(0, 3, 1, 2))

        if not result:
            # If no watermark found, return default values
            return (0, 0, "Failed")
        
        best_x, best_y, final_result = result
        # Convert the numeric final_result to a string with formatting
        return (best_x, best_y, f"Chance of watermark: {final_result:.4f}")


# ------------------------------------------------------------------------------
# Node 2: RemoveWatermarkNode
# ------------------------------------------------------------------------------
class RemoveWatermarkNode:
    """
    This node takes a batch of images and the (x, y) positions of the watermark, then removes it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ), 
            },
            "optional": {
                "x": ("INT", {"forceInput":True}),
                "y": ("INT", {"forceInput":True}),
                "nudge_x": ("INT", {"default": 0, "min": -999999, "max": 999999, "step": 1}),
                "nudge_y": ("INT", {"default": 0, "min": -999999, "max": 999999, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images_corrected",)
    FUNCTION = "remove_watermark"
    CATEGORY = "Video/Watermarks"

    def remove_watermark(self, images, x=0, y=0, nudge_x=0, nudge_y=0):
        """
        Args:
            images (list of PIL.Image): Batch of input images.
            x (int): x-coordinate of the watermark.
            y (int): y-coordinate of the watermark.
            nudge_x (int): Additional x nudge value.
            nudge_y (int): Additional y nudge value.

        Returns:
            Tuple[list of PIL.Image]: Batch of corrected images without the watermark.
        """
        x += nudge_x
        y += nudge_y

        # Call your existing watermark removal function
        corrected_frames = find_watermarks.remove_watermark_batch(images.permute(0, 3, 1, 2), x, y)

        return (corrected_frames.permute(0, 2, 3, 1),)