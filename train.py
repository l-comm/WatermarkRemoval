import os
from pathlib import Path
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.brightness_predictor import BrightnessToAlphaBeta
from dataset import WatermarkDataset
from models.watermark_detector import WatermarkDetector
import cv2
import numpy as np
from torchvision import transforms as T

device = 'cuda:0'

current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
checkpoint_dir = os.path.join(current_dir, "checkpoints")
assets_dir = os.path.join(current_dir, "assets")
debug_dir = os.path.join(current_dir, "debug")
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')

# Load the watermark detection model
# model_ft, base_transforms = get_convnext_model('convnext-tiny')
weights = torch.load(os.path.join(checkpoint_dir, "convnext-tiny_watermarks_detector.pth"), map_location=device)

# Directory for results
if os.path.exists(debug_dir):
    shutil.rmtree(debug_dir)
os.makedirs(debug_dir, exist_ok=True)

predictor = WatermarkDetector(weights, device=device)

# Known ROI size (must match border_map, body_map)
ROI_WIDTH = 330
ROI_HEIGHT = 60
FRAMES_PER_SAMPLE = 40

dataset = WatermarkDataset(
    csv_path=os.path.join(current_dir,'watermark_locations.csv'), 
    border_map_path=os.path.join(assets_dir, 'shutter_border_sm.png'),
    body_map_path=os.path.join(assets_dir, 'shutter_text_sm.png'),
    roi_width=ROI_WIDTH,
    roi_height=ROI_HEIGHT,
    frames_per_sample=FRAMES_PER_SAMPLE
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

alpha_beta_model = BrightnessToAlphaBeta().to(device)
optimizer = optim.Adam(alpha_beta_model.parameters(), lr=1e-1)

# Create a ReduceLROnPlateau scheduler that reduces the LR by factor 0.1 after 10 epochs of no improvement
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    alpha_beta_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 0

num_epochs = 1000

for epoch in range(start_epoch, num_epochs):
    total_loss = 0.0
    num_batches = 0
    batch_idx = 0
    for batch in dataloader:
        brightness_per_frame = batch['brightness_per_frame'].squeeze(0).to(device)  # [F]
        roi_gray = batch['roi_gray'].squeeze(0).to(device)                           # [F, roiH, roiW]
        roi_color = batch['roi_color'].squeeze(0).to(device)                         # [F, roiH, roiW, C]
        border_map = batch['border_map'].squeeze(0).to(device)                       # [roiH, roiW]
        body_map = batch['body_map'].squeeze(0).to(device)                           # [roiH, roiW]
        original_color_frames = batch['original_color_frames'].squeeze(0).to(device) # [F,H,W,C]
        x = batch['x'].item()
        y = batch['y'].item()
        
        # Predict alpha,beta for each frame
        # brightness_per_frame: [F]
        # alpha_beta_model expects input of shape [F,1]
        # print("brightness_per_frame shape:", original_color_frames.shape)
        alpha_beta = alpha_beta_model(brightness_per_frame.unsqueeze(1)) # [F,2]
        alpha = alpha_beta[:,0]  # [F]
        beta = alpha_beta[:,1]   # [F]
        
        # We'll compute probability per frame and average for the loss
        frame_probabilities = []
        
        for f_idx in range(roi_gray.shape[0]):
            single_roi_gray = roi_gray[f_idx]       # [roiH, roiW]
            single_roi_color = roi_color[f_idx]     # [roiH, roiW, C]
            single_original = original_color_frames[f_idx] # [H,W,C]

            # Apply correction: corrected_roi = roi_gray - alpha*border_map + beta*body_map
            # alpha and beta are scalars per frame, so index with [f_idx]
            # corrected_roi = single_roi_gray - 0.5*border_map + 0.5*body_map
            corrected_roi = single_roi_gray - alpha[f_idx]*border_map + beta[f_idx]*body_map
            corrected_roi = torch.clamp(corrected_roi, 0, 255)
            
            # difference = corrected_roi - roi_gray
            difference = corrected_roi - single_roi_gray
            
            difference_3d = difference.unsqueeze(-1)  # [roiH, roiW, 1]
            roi_color_corrected = single_roi_color + difference_3d
            roi_color_corrected = torch.clamp(roi_color_corrected, 0, 255)
            
            # Insert corrected ROI back into the original frame
            final_corrected_frame = single_original.clone()
            final_corrected_frame[y:y+ROI_HEIGHT, x:x+ROI_WIDTH, :] = roi_color_corrected
            
            # Predict watermark probability for this corrected frame
            # final_corrected_frame = final_corrected_frame[..., [2, 1, 0]]  # Reorder channels from BGR to RGB
            probability = predictor.predict_tensor(final_corrected_frame) # Tensor scalar
            frame_probabilities.append(probability)

            debug_image = final_corrected_frame.detach().cpu().numpy().astype(np.uint8)
            # cv2.imwrite(os.path.join(debug_dir, f"final_corrected_frame_epoch_{epoch}_batch_{batch_idx}_frame_{f_idx}.png"), cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
            # print(f"{batch_idx} {f_idx}")
            if (f_idx == 0 or f_idx == 39):
                cv2.imwrite(os.path.join(debug_dir, f"final_corrected_frame_batch_{batch_idx}_frame_{f_idx}.png"), debug_image)
            batch_idx += 1
        
        # Average over frames to get the final loss
        # print(f"frame probs {frame_probabilities}")
        frame_probabilities = torch.stack(frame_probabilities)  # [F]
        loss = frame_probabilities.mean()
        total_loss += loss.item()
        num_batches += 1
        # print(f"LOSS {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    scheduler.step(avg_loss)
    current_lr = scheduler.get_last_lr()
    print(f"Current LR: {current_lr}")
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    print(f"Weight:{alpha_beta_model.fc.weight.data} Bias: {alpha_beta_model.fc.bias.data}")

    # Save the checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': alpha_beta_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
