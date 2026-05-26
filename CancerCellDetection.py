from torchvision import models
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd

cap = cv2.VideoCapture(0)
yolo_model = YOLO("yolo11m.pt")

while True:
    img, frame = cap.read()
    if img == False::
        break
    
    results = yolo_model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

    class CancerCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()

#model = models.resnet18(weights="DEFAULT")

#model.eval()


