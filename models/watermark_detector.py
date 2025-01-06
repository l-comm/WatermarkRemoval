# watermark_detector.py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from .convnext import ConvNeXt

class WatermarkDetector:
    def __init__(self, weights, device='cpu'):
        # Initialize model
        self.model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        self.model.head = nn.Sequential( 
            nn.Linear(in_features=768, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=2),
        )
        self.model.load_state_dict(weights)

        # Transforms for a SINGLE image of shape [3, H, W]
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def predict_tensor(self, image_tensor: torch.Tensor):
        """
        Predicts the probability of a watermark for either:
          - A single image:  shape [3, H, W]
          - A batch of images: shape [B, 3, H, W]

        Returns:
          - float if single image
          - 1D Tensor of shape [B] if batched
        """
        # Move to the correct device (if not already)
        image_tensor = image_tensor.to(self.device)

        # Case 1: Single image [3, H, W]
        if image_tensor.dim() == 3:
            # Apply transforms
            # shape remains [3, H, W] => after transforms => [3, 256, 256]
            image_tensor = self.transforms(image_tensor)
            # Add batch dimension => [1, 3, 256, 256]
            image_tensor = image_tensor.unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)  # [1, 2]
                # Return probability of "class 1"
                return probabilities[0, 1].item()

        # Case 2: Batch of images [B, 3, H, W]
        elif image_tensor.dim() == 4:
            B = image_tensor.shape[0]
            # Apply transforms to each item in the batch
            transformed_list = []
            for i in range(B):
                # transforms expects [3, H, W]
                one_img = image_tensor[i]
                one_img = self.transforms(one_img)
                transformed_list.append(one_img)
            # Stack back => [B, 3, 256, 256]
            image_tensor = torch.stack(transformed_list, dim=0)

            with torch.no_grad():
                outputs = self.model(image_tensor)  # => [B, 2]
                probabilities = F.softmax(outputs, dim=1)  # => [B, 2]
                # Return [B] probabilities for class=1
                return probabilities[:, 1]

        else:
            raise ValueError(
                f"predict_tensor expects [3,H,W] or [B,3,H,W], got shape={image_tensor.shape}"
            )
