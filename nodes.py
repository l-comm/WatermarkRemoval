# nodes.py

import torch
from . import find_watermarks

# ------------------------------------------------------------------------------
# Node 1: FindWatermarkNode
# ------------------------------------------------------------------------------
class FindWatermarkNode:
    """
    This node takes a batch of video frames (as a torch.Tensor) and returns:
      - The x and y coordinates of the detected watermark
      - The watermark detection score as a string
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixel_values": ("TENSOR", ),  # ComfyUI custom type specifying a torch.Tensor
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("x", "y", "score")
    FUNCTION = "find_watermark"             # Method name below
    CATEGORY = "Video/Watermarks"           # Appears under this category in the UI

    def find_watermark(self, pixel_values):
        # Call your existing function from find_watermarks.py
        result = find_watermarks.find_watermark_tensor(pixel_values)

        if not result:
            # If no watermark found, return a default
            return (0, 0, "No watermark found")
        
        best_x, best_y, final_result = result
        # Convert the numeric final_result to a string
        return (best_x, best_y, f"{final_result:.4f}")


# ------------------------------------------------------------------------------
# Node 2: RemoveWatermarkNode
# ------------------------------------------------------------------------------
class RemoveWatermarkNode:
    """
    This node takes a batch of video frames (as a torch.Tensor) and
    the (x, y) positions of the watermark, then removes it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixel_values": ("TENSOR", ),
                "x": ("INT", ),
                "y": ("INT", ),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("pixel_values_corrected",)
    FUNCTION = "remove_watermark"
    CATEGORY = "Video/Watermarks"

    def remove_watermark(self, pixel_values, x, y):
        # Call your existing function from find_watermarks.py
        corrected_frames = find_watermarks.remove_watermark_batch(pixel_values, x, y)
        return (corrected_frames,)
