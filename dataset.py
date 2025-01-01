import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class WatermarkDataset(Dataset):
    def __init__(self, csv_path, border_map_path, body_map_path, roi_width, roi_height, frames_per_sample=4):
        self.df = pd.read_csv(csv_path)
        
        border_map = cv2.imread(border_map_path, cv2.IMREAD_GRAYSCALE)
        body_map = cv2.imread(body_map_path, cv2.IMREAD_GRAYSCALE)
        if border_map is None or body_map is None:
            raise FileNotFoundError("border_map or body_map not found")
        
        if border_map.shape[0] != roi_height or border_map.shape[1] != roi_width:
            raise ValueError("border_map size does not match ROI expected size.")
        if body_map.shape[0] != roi_height or body_map.shape[1] != roi_width:
            raise ValueError("body_map size does not match ROI expected size.")
        
        self.border_map = torch.from_numpy(border_map.astype(np.float32))
        self.body_map = torch.from_numpy(body_map.astype(np.float32))
        
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.frames_per_sample = frames_per_sample

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['video_path']
        x = int(row['x'])
        y = int(row['y'])
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not read video {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine how many frames we will actually read
        frames_to_read = min(self.frames_per_sample, frame_count)
        
        # Randomly select frame indices (unique, sorted)
        frame_indices = np.random.choice(frame_count, frames_to_read, replace=False)
        frame_indices.sort()
        
        frames = []
        roi_colors = []
        roi_grays = []
        
        # Read selected frames
        for frame_index in frame_indices:
            # Seek to the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                # If we fail to read a frame, skip it or raise an error
                # Here we choose to raise an error
                raise ValueError(f"Could not read frame {frame_index} from {video_path}")
            
            # Extract ROI
            if (y + self.roi_height) > frame.shape[0] or (x + self.roi_width) > frame.shape[1]:
                raise ValueError(f"ROI exceeds frame boundaries. Frame shape: {frame.shape}")
            
            roi_color = frame[y:y+self.roi_height, x:x+self.roi_width, :]
            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            
            roi_colors.append(roi_color.astype(np.float32))
            roi_grays.append(roi_gray.astype(np.float32))
            frames.append(frame.astype(np.float32))
        
        cap.release()
        
        if len(roi_colors) == 0:
            raise ValueError(f"No frames read from {video_path}")
        
        full_frames_tensor = torch.from_numpy(np.stack(frames, axis=0))      # [F,H,W,C]
        roi_color_tensor = torch.from_numpy(np.stack(roi_colors, axis=0))    # [F,roiH,roiW,C]
        roi_gray_tensor = torch.from_numpy(np.stack(roi_grays, axis=0))      # [F,roiH,roiW]
        
        # Compute brightness per frame
        brightness_per_frame = roi_gray_tensor.mean(dim=(1,2))/255.  # shape [F]

        return {
            'brightness_per_frame': brightness_per_frame,  # [F]
            'roi_gray': roi_gray_tensor,                   # [F, roiH, roiW]
            'roi_color': roi_color_tensor,                 # [F, roiH, roiW, C]
            'border_map': self.border_map,                 # [roiH, roiW]
            'body_map': self.body_map,                     # [roiH, roiW]
            'x': x,
            'y': y,
            'original_color_frames': full_frames_tensor    # [F,H,W,C]
        }
