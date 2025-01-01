import cv2
import torch
from torchvision import models, transforms
import torch.nn as nn
from .convnext import ConvNeXt
import torch.nn.functional as F

class WatermarkDetector:
    def __init__(self, weights, device='cpu'):
        self.model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        self.model.head = nn.Sequential( 
            nn.Linear(in_features=768, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=2),
        )
        self.model.load_state_dict(weights)
        self.transforms = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = device
        self.model.to(self.device)
    
    def predict_pil(self, pil_image):
        # Ensure the image is in RGB format
        pil_image = pil_image.convert("RGB")
        
        # Apply the transformations and add a batch dimension
        input_tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)
        
        # Perform inference without tracking gradients
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Extract the probability for the 'watermarked' class
        watermarked_prob = probabilities[0, 1]
        
        return watermarked_prob
        
    def predict_tensor(self, image_tensor):

        image_tensor = image_tensor.permute(2,0,1).float()/255.0
        image_tensor = self.transforms(image_tensor).unsqueeze(0).to(self.device)

        outputs = self.model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        return probabilities[0, 1]
    
    def predict_cv(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(rgb_image)
        return self.predict_tensor(rgb_tensor)
