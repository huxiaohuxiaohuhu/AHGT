import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os

save_path = './model/faster_rcnn/fasterrcnn_resnet50_fpn.pth'

if not os.path.exists(save_path):
    print("Downloading Faster R-CNN model...")
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
else:
    print(f"Model already exists at {save_path}")
