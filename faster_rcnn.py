import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import RoIAlign
from collections import OrderedDict

import os

class FasterRCNNFeatureExtractor(nn.Module):
    def __init__(self, device, num_keep=49, model_path='./model/faster_rcnn/fasterrcnn_resnet50_fpn.pth'):
        super().__init__()
        # Load pretrained Faster R-CNN
        # If local path exists, load from there, otherwise download (but set pretrained=False first to avoid download if we load state dict)
        
        if os.path.exists(model_path):
            print(f"Loading Faster R-CNN from local path: {model_path}")
            self.model = fasterrcnn_resnet50_fpn(pretrained=False)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("Downloading Faster R-CNN...")
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            
        self.model.eval()
        self.model.to(device)
        self.num_keep = num_keep
        self.device = device
        
        # FPN output channels is 256
        self.out_dim = 256 
        
        # RoIAlign to extract features from backbone
        # We use 7x7 output size to match the spatial resolution if we were to pool
        # But here we want a vector per box. 
        # If we do 1x1, we get 256 dim vector.
        # If we do 7x7, we get 256x7x7 vector which is huge.
        # Let's do 1x1 to keep it manageable, or 2x2.
        # Wait, ResNet152 in original code gave 2048 dim.
        # If we use 7x7 from FPN (256 channels), we get 256*49 = 12544 dim.
        # We can add a linear layer to project it to 2048.
        
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0/32, sampling_ratio=2)
        # Note: spatial_scale needs to match the feature map. 
        # FPN has multiple scales. 
        
        # To simplify, let's just use the backbone's top layer feature map directly 
        # if we want grid features, OR use the detections.
        
        # Let's implement "Top K Boxes" approach.

    def train(self, mode=True):
        # Override train so that the internal model always stays in eval mode
        # This is necessary because we are using it as a feature extractor without targets
        super().train(mode)
        self.model.eval()
        return self
        
    def forward(self, images):
        # images: Tensor [N, 3, H, W]
        # Faster R-CNN expects List[Tensor]
        
        batch_size = images.size(0)

        # 1. Run model to get detections
        # We need to wrap images in list
        image_list = [img for img in images]
        
        # Ensure model is in eval mode just in case
        self.model.eval()
        
        # Use torch.no_grad() to save memory since we are only doing inference
        with torch.no_grad():
            # This returns detections
            detections = self.model(image_list)
            
            # This returns backbone features
            # We need to access backbone directly because model() doesn't return features
            backbone_features = self.model.backbone(images)
            
        # backbone_features is a dict: '0', '1', '2', '3' (FPN levels)
        # '0' is the highest resolution (stride 4), '3' is lowest (stride 32).
        
        # Let's go with the "Backbone Grid" approach for stability.
        # It uses the FPN backbone of Faster R-CNN.
        
        feat = backbone_features['0'] # [N, 256, H', W']
        # Pool to 7x7
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (7, 7)) # [N, 256, 7, 7]
        
        # Global average pooling for x and fc simulation
        global_feat = torch.mean(feat, dim=[2, 3]) # [N, 256]
        
        # Normalize features to avoid exploding gradients
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        global_feat = torch.nn.functional.normalize(global_feat, p=2, dim=1)
        
        # Return 3 values to match ResNet interface: x, fc, att
        # att should be [N, 256, 7, 7] (or similar that can be viewed as [N, 256, 49])
        return global_feat, global_feat, feat

