# find_watermarks.py

import logging
import requests
from rich.logging import RichHandler
from rich.console import Console
import os
from pathlib import Path
import csv
import cv2
from tqdm import tqdm
import imageio
import numpy as np
import torch
import torchaudio
from .models.brightness_predictor import BrightnessToAlphaBeta
from .models.watermark_detector import WatermarkDetector
from collections import Counter
import torch.nn.functional as F

console = Console(highlight=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            omit_repeated_times=False,
        ),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# Global models
predictor = None
b2ab = None

current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
checkpoint_dir = os.path.join(current_dir, "checkpoints")
assets_dir = os.path.join(current_dir, "assets")

# Load your templates/masks once
border_map = cv2.imread(os.path.join(assets_dir, 'shutter_border_sm.png'), cv2.IMREAD_GRAYSCALE)
body_map = cv2.imread(os.path.join(assets_dir, 'shutter_text_sm.png'), cv2.IMREAD_GRAYSCALE)
border_tensor = torch.from_numpy(border_map).float() / 255.0  # Shape: [H, W]
body_tensor = torch.from_numpy(body_map).float() / 255.0      # Shape: [H, W]
border_tensor = border_tensor.to('cuda')
body_tensor = body_tensor.to('cuda')

template_tensor = body_tensor
mask_tensor = (template_tensor > 6.0/255.0).float()

# Directory for results
results_dir = os.path.join(current_dir,'watermark_results')
os.makedirs(results_dir, exist_ok=True)

def load_models():
    global predictor, b2ab
    WATERMARK_DETECTOR_URL = "https://huggingface.co/boomb0om/watermark-detectors/resolve/main/convnext-tiny_watermarks_detector.pth"
    watermark_model_location = Path(os.path.join(checkpoint_dir, "convnext-tiny_watermarks_detector.pth"))
    brightness_model_location = Path(os.path.join(checkpoint_dir, 'brightness_predictor.pth'))
    
    ensure_file_exists(watermark_model_location, WATERMARK_DETECTOR_URL)
    device = 'cuda'
    if predictor is None:
        weights = torch.load(watermark_model_location, device)
        predictor = WatermarkDetector(weights, device=device)
        predictor.model.to(device)
        predictor.model.eval()
    if b2ab is None:
        b2ab = BrightnessToAlphaBeta()
        b2ab.load_state_dict(torch.load(brightness_model_location)['model_state_dict'])
        b2ab.to(device)
        b2ab.eval()

def download_file(url, dest_path):
    """
    Downloads a file from the specified URL to the destination path with a progress bar.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f, tqdm(
            desc=f"Downloading {dest_path.name}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        logger.info(f"Downloaded {dest_path.name} successfully.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        raise

def ensure_file_exists(file_path, url):
    """
    Ensures that the file exists locally; downloads it from the URL if it does not.
    """
    if not file_path.exists():
        logger.info(f"{file_path.name} not found. Downloading from HuggingFace...")
        download_file(url, file_path)

def save_pixel_values_as_video(pixel_values, output_path):
    """
    Save pixel values as a video file with a target FPS of 24.

    Parameters:
    pixel_values (Tensor): Tensor containing pixel values with shape (N, C, H, W),
                           where N is the number of frames, C is the number of channels,
                           H is the height, and W is the width.
    output_path (str): The output path for the video file.
    """
    target_fps = 24
    writer = imageio.get_writer(output_path, fps=target_fps)

    try:
        for frame in pixel_values.cpu().numpy():
            # Convert the frame from (C, H, W) to (H, W, C)
            frame = np.transpose(frame, (1, 2, 0))
            writer.append_data((frame * 255).astype(np.uint8))
    finally:
        writer.close()

    logger.info(f"Video saved as {output_path}")

def match_template(
    image: torch.Tensor,
    template: torch.Tensor,
    mask: torch.Tensor = None,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    High-performance PyTorch implementation of cv2.matchTemplate 
    with method=cv2.TM_CCOEFF_NORMED, using 2D convolutions on GPU.

    Args:
        image:    Float tensor [H, W] or [1, H, W], grayscale image.
        template: Float tensor [th, tw] or [1, th, tw], grayscale template.
        mask:     Float tensor [th, tw] or [1, th, tw], binary {0,1}, optional.
        eps:      Small constant to avoid division by zero.

    Returns:
        result:   A float tensor [H - th + 1, W - tw + 1] with 
                  the normalized cross-correlation scores.
    """
    # -------------------------------------------------------------------------
    # Preprocessing and shape handling
    # -------------------------------------------------------------------------
    # Ensure image shape is [1, H, W]
    if image.dim() == 2:
        image = image.unsqueeze(0)
    elif image.dim() == 3 and image.size(0) == 1:
        pass  # shape is already [1, H, W]
    else:
        raise ValueError("image must be [H, W] or [1, H, W] with grayscale data.")
    _, H, W = image.shape

    # Ensure template shape is [th, tw]
    if template.dim() == 2:
        pass  # shape is already [th, tw]
    elif template.dim() == 3 and template.size(0) == 1:
        template = template.squeeze(0)
    else:
        raise ValueError("template must be [th, tw] or [1, th, tw].")
    th, tw = template.shape

    # Handle mask
    if mask is None:
        # Use a mask of all ones
        mask = torch.ones_like(template)
    else:
        if mask.dim() == 3 and mask.size(0) == 1:
            mask = mask.squeeze(0)
        if mask.shape != template.shape:
            raise ValueError("mask must have the same spatial shape as template.")
    mask = (mask > 0).float()  # ensure binary in {0,1}

    # Move everything to the same device & float (in case user wants GPU)
    device = image.device
    image = image.float().to(device)
    template = template.float().to(device)
    mask = mask.float().to(device)

    # -------------------------------------------------------------------------
    # Compute template statistics (masked)
    # -------------------------------------------------------------------------
    valid_count_template = mask.sum()
    if valid_count_template < 1:
        raise ValueError("Template mask has zero valid pixels.")

    t_mean = (template * mask).sum() / valid_count_template
    # Centered template: (T - mean) * mask
    t_centered = (template - t_mean) * mask
    # Sum of squares of template
    t_sum_sq = (t_centered ** 2).sum()

    # -------------------------------------------------------------------------
    # Prepare convolution kernels and image for conv2d
    # -------------------------------------------------------------------------
    # We will treat image as [N=1, C=1, H, W] so that F.conv2d can work directly.
    # Convert template, mask into [out_channels=1, in_channels=1, th, tw]
    image_4d = image.unsqueeze(0)  # [1, 1, H, W]
    t_centered_4d = t_centered.unsqueeze(0).unsqueeze(0)  # [1,1,th,tw]
    mask_4d = mask.unsqueeze(0).unsqueeze(0)              # [1,1,th,tw]

    # -------------------------------------------------------------------------
    # Convolution for cross-term: sum over patch [ I * (T - mean(T)) ]
    # -------------------------------------------------------------------------
    # cross_map shape => [1, 1, H - th + 1, W - tw + 1]
    cross_map = F.conv2d(image_4d, t_centered_4d)

    # -------------------------------------------------------------------------
    # Convolution for sums in the image patch (with mask):
    #   i_sum: sum over patch [ I * mask ]
    #   i2_sum: sum over patch [ I^2 * mask ]
    # -------------------------------------------------------------------------
    i_sum = F.conv2d(image_4d, mask_4d)         # sum of I * mask
    i2_sum = F.conv2d(image_4d ** 2, mask_4d)   # sum of I^2 * mask

    # -------------------------------------------------------------------------
    # Mean and variance for each patch
    #
    # i_mean = i_sum / valid_count_template
    # var(I) = sum((I - i_mean)^2 * mask)
    #         = sum(I^2 * mask) - i_sum^2 / valid_count_template
    # -------------------------------------------------------------------------
    i_mean = i_sum / valid_count_template
    i_var = i2_sum - (i_sum ** 2) / valid_count_template

    # -------------------------------------------------------------------------
    # Denominator = sqrt( sum((T - t_mean)^2) * var(I) ), broadcast
    # -------------------------------------------------------------------------
    denominator = torch.sqrt(t_sum_sq * i_var + eps)

    # -------------------------------------------------------------------------
    # Normalized cross-correlation map
    # -------------------------------------------------------------------------
    # shape => [1, 1, H-th+1, W-tw+1]
    result_map = cross_map / denominator

    # Squeeze [1, 1, ...] -> [H-th+1, W-tw+1]
    return result_map.squeeze(0).squeeze(0)

def remove_watermark_batch(
    batch_frames: torch.Tensor,
    x: int,
    y: int
) -> torch.Tensor:
    """
    Removes watermark from a batch of video frames.

    Args:
        batch_frames (torch.Tensor): Input video frames of shape (b, c, H, W).
        x (int): The x-coordinate of the top-left corner of the ROI.
        y (int): The y-coordinate of the top-left corner of the ROI.

    Returns:
        torch.Tensor: Corrected video frames of shape (b, c, H, W).
    """
    with torch.no_grad():
        global body_tensor, border_tensor, predictor, b2ab
        load_models()
        # Ensure batch_frames are in float for processing
        device = 'cuda'
        batch_frames = batch_frames.to(device, dtype=torch.float32)

        b, c, H, W = batch_frames.shape
        h, w = border_tensor.shape

        # Move maps to the same device as batch_frames
        border_tensor = border_tensor.to(batch_frames.device)
        body_tensor = body_tensor.to(batch_frames.device)

        scale_r = 0.2989
        scale_g = 0.5870
        scale_b = 0.1140

        # Convert to grayscale: 0.2989 * R + 0.5870 * G + 0.1140 * B
        grayscale = scale_r * batch_frames[:, 0, :, :] + scale_g * batch_frames[:, 1, :, :] + scale_b * batch_frames[:, 2, :, :]  # Shape: (b, H, W)

        # Extract ROI from grayscale
        roi_gray = grayscale[:, y:y+h, x:x+w]  # Shape: (b, h, w)

        # Compute brightness: mean over h and w, normalized
        brightness = roi_gray.mean(dim=[1, 2])  # Shape: (b,)

        # Reshape brightness for model input
        brightness = brightness.unsqueeze(1)  # Shape: (b, 1)

        # Pass brightness through the model to get alpha and beta
        alpha_beta = b2ab(brightness)  # Shape: (b, 2)

        # Split alpha and beta and reshape to (b, 1, 1) for broadcasting
        alpha = alpha_beta[:, 0].view(b, 1, 1)  # Shape: (b, 1, 1)
        beta = alpha_beta[:, 1].view(b, 1, 1)   # Shape: (b, 1, 1)

        # Expand border_tensor and body_tensor to match roi_gray dimensions
        # Shape: (1, h, w) -> (b, h, w)
        border_tensor_expanded = border_tensor.unsqueeze(0).expand(b, -1, -1)
        body_tensor_expanded = body_tensor.unsqueeze(0).expand(b, -1, -1)

        # logger.info(f"roi gray {roi_gray.shape} - alpha {alpha.shape} - border {border_tensor_expanded.shape}")

        # Compute corrected ROI in grayscale
        corrected_roi = roi_gray - (alpha * border_tensor_expanded) + (beta * body_tensor_expanded)

        # Clamp corrected ROI to [0, 1]
        corrected_roi = torch.clamp(corrected_roi, 0.0, 1.0)

        # Compute the difference
        difference = corrected_roi - roi_gray  # Shape: (b, h, w)

        # Apply scaled differences
        # difference_3d = torch.stack([
        #     difference * scale_r,
        #     difference * scale_g,
        #     difference * scale_b
        # ], dim=1)  # Shape: (b, 3, h, w)
        # Expand difference to 3 channels
        difference_3d = difference.unsqueeze(1).repeat(1, 3, 1, 1)  # Shape: (b, 3, h, w)

        # Extract color ROI from original frames
        roi_color = batch_frames[:, :, y:y+h, x:x+w]  # Shape: (b, 3, h, w)

        # Apply the difference
        roi_color_corrected = roi_color + difference_3d  # Shape: (b, 3, h, w)

        # Clamp the corrected color ROI to [0, 1]
        roi_color_corrected = torch.clamp(roi_color_corrected, 0.0, 1.0)

        # Create a copy of the original frames to avoid modifying them in-place
        corrected_frames = batch_frames.clone()

        # Replace the ROI in the copied frames with the corrected ROI
        corrected_frames[:, :, y:y+h, x:x+w] = roi_color_corrected

        # If original frames were in integer type, convert back
        if corrected_frames.dtype != batch_frames.dtype:
            corrected_frames = corrected_frames.to(batch_frames.dtype)

    return corrected_frames

def rgb_to_grayscale_torch(image: torch.Tensor) -> torch.Tensor:
    """
    image: [3, H, W]  (float in 0..1 or 0..255)
    Returns: [H, W]   grayscale
    """
    # Weighted sum of RGB
    # If your original images were BGR in OpenCV, be sure to correct order as needed.
    # But typically with PyTorch we store color as RGB.
    r = image[0]
    g = image[1]
    b = image[2]
    gray = 0.299*r + 0.587*g + 0.114*b
    return gray


def find_watermark_frame_tensor(original_color_torch: torch.Tensor, result_cutoff=0.9):
    """
    Replaces find_watermark_frame_cv() but uses only PyTorch on GPU.
    original_color_torch: shape [3, H, W], float32, device='cuda'
    """

    load_models()

    # Convert color -> grayscale
    # If your original images were actually BGR, swap channels accordingly
    original_gray_torch = rgb_to_grayscale_torch(original_color_torch)

    # 1) Template match in PyTorch
    res = match_template(
        original_gray_torch,  # shape [H, W]
        template_tensor,       # shape [th, tw]
        mask_tensor            # shape [th, tw], optional
    )
    
    # 2) Replicate minMaxLoc
    min_val, max_val, min_loc, max_loc = min_max_loc_torch(res)

    # We'll search around min_loc in a small window
    # In OpenCV, min_loc is (x, y).  That means:
    init_x, init_y = min_loc  # (x, y)

    h, w = border_tensor.shape[-2:]  # border_map size
    search_range = 2

    best_prob = float('inf')
    best_x = 0
    best_y = 0

    original_color_torch = original_color_torch.unsqueeze(0)
    # 3) Search around initial location
    for dx in range(-search_range, search_range+1):
        for dy in range(-search_range, search_range+1):
            x = init_x + dx
            y = init_y + dy

            # "Remove" watermark in a pure PyTorch manner
            # logger.info(f"original_color_torch {original_color_torch.shape}")
            corrected_image_torch = remove_watermark_batch(original_color_torch, x, y)
            
            # predictor on GPU
            # Suppose your predictor takes a [C, H, W] float32 in [0..1]
            # and returns a float. (Adjust as needed.)
            result_prob = predictor.predict_tensor(corrected_image_torch[0])
            if result_prob < best_prob:
                best_prob = result_prob
                best_x = x
                best_y = y

    # 4) Compute final corrected image
    # logger.info(f"original_color_torch 2 {original_color_torch.shape}")
    final_corrected_image_torch = remove_watermark_batch(original_color_torch, best_x, best_y)
    final_result = predictor.predict_tensor(final_corrected_image_torch[0])
    print(f"final {final_result}")
    if final_result > result_cutoff:
        return None

    return best_x, best_y, float(final_result)

def min_max_loc_torch(matrix: torch.Tensor):
    """
    Replicates cv2.minMaxLoc for a 2D PyTorch tensor.

    Returns:
        min_val, max_val (float)
        min_loc, max_loc ((x, y) as in OpenCV)
    """
    # Flattened min / max
    min_val = matrix.min()
    max_val = matrix.max()
    min_idx = torch.argmin(matrix)
    max_idx = torch.argmax(matrix)

    # Convert flat index -> 2D (y, x)
    h, w = matrix.shape
    min_y, min_x = divmod(min_idx.item(), w)
    max_y, max_x = divmod(max_idx.item(), w)

    # OpenCV returns (x, y)
    return (min_val.item(),
            max_val.item(),
            (min_x, min_y),
            (max_x, max_y))

def find_watermark_tensor(pixel_values, frame_range=range(0, 24, 5), result_cutoff=0.9):
    """
    Replaces find_watermark_tensor but does all processing in PyTorch on GPU.
    
    pixel_values: shape [num_frames, 3, H, W], float32, device='cuda'
    frame_range:  which frames to process
    """
    device = 'cuda'
    pixel_values = pixel_values.to(device)

    total_frames = pixel_values.shape[0]
    frame_positions = []

    for frame_idx in frame_range:
        if frame_idx >= total_frames:
            break

        # We no longer convert to OpenCV image
        # We directly use the PyTorch tensor
        frame_torch = pixel_values[frame_idx]  # shape [3, H, W]

        res = find_watermark_frame_tensor(frame_torch, result_cutoff)
        if not res:
            logger.error(f"{frame_idx} Failed to find")
            continue

        best_x, best_y, final_result = res
        frame_positions.append((best_x, best_y, final_result))

    # If no frames succeeded
    if not frame_positions:
        return None

    # Majority vote on (x, y) location
    positions = [(fp[0], fp[1]) for fp in frame_positions]
    position_to_result = {(fp[0], fp[1]): fp[2] for fp in frame_positions}
    position_counts = Counter(positions)
    # Most common
    most_common_position, _ = position_counts.most_common(1)[0]
    final_result = position_to_result[most_common_position]

    logger.info(f"MOST COMMON {most_common_position}, {final_result}")

    return most_common_position[0], most_common_position[1], final_result

def find_watermark_test_tensor():
    # Directory with videos
    video_dir = os.path.join(current_dir, 'watermarked_videos')
    video_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
    
    sample_n_frames = 24
    target_fps = 24
    fit_to = 336
    final_x, final_y = None, None
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)

        logger.info(f"Initializing StreamReader for video URL: {video_path}")
        stream_reader = torchaudio.io.StreamReader(video_path)
        stream_reader.add_video_stream(
            frames_per_chunk=sample_n_frames,
            filter_desc=(
                f"fps={target_fps},"
                f"scale='if(gt(iw,ih),{fit_to}*iw/ih,{fit_to}):if(gt(iw,ih),{fit_to},{fit_to}*ih/iw)',"
                "crop=w=floor(iw/8)*8:h=floor(ih/8)*8:x=(iw-floor(iw/8)*8)/2:y=(ih-floor(ih/8)*8)/2,"
                "format=pix_fmts=rgb24"
            )
        )
        stream_reader.seek(0)
        stream_reader.fill_buffer()
        (frames,) = stream_reader.pop_chunks()
        pixel_values = frames.float() / 255.0
        
        logger.info(f"Pixel values {pixel_values.shape}")
        res = find_watermark_tensor(pixel_values)
        if not res:
            continue
        final_x, final_y = res
        logger.info(f"Found watermark position - {final_x} {final_y}")

        pixel_values = remove_watermark_batch(pixel_values, final_x, final_y)
        logger.info(f"Removed watermark")

        base_name = os.path.splitext(video_file)[0]
        corrected_path = os.path.join(results_dir, f"{base_name}_corrected.mp4")
        save_pixel_values_as_video(pixel_values, corrected_path)

if __name__ == "__main__":
    find_watermark_test_tensor()