from ultralytics import YOLO
from gradcam import GradCAM
import cv2
import torch
from torch import nn



img_path = "6_jpg.rf.b3dbdcd56ecce6980a163a59431dad5d.jpg"

yolo_model = YOLO("best.pt")
results = yolo_model(img_path)
annotated_frame = results[0].plot()
cv2.imshow("YOLO", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

class CancerCNN(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)#16 output layers
      self.relu =nn.ReLU()
      self.pool = nn.MaxPool2d(2)#after each convolution, we apply max pooling to reduce spatial dimensions
      self.conv2 = nn.Conv2d(16,32, kernel_size=3, padding=1)#32 output layers
      self.conv3 = nn.Conv2d(32,64, kernel_size=3, padding=1)#64 output layers
      self.flatten = nn.Flatten()#converts 3D tensor to 1D tensor
      self.fc1 = nn.Linear(4096, 256)#64 of 8*8 pixels and 256 output layers
      self.fc2 = nn.Linear(256, 2)#2 output layers, Benign and Malignant

   def forward(self, x):
      x = self.pool(self.relu(self.conv1(x))) #Takes image, runs through conv1, applies ReLU to remove negatives, shrink it in half with pool making image half the size
      x = self.pool(self.relu(self.conv2(x)))
      x = self.pool(self.relu(self.conv3(x)))
      x = self.flatten(x)#converts a 3D tensor to a 1D tensor
      x = self.relu(self.fc1(x))#applies ReLu to remove negatives. Takes 4096 numbers and compresses them down to 256 features.
      x = self.fc2(x)#2 output layers, Benign and Malignant
      return x

  



#cnn_model = CancerCNN()
#cnn_model.eval()


    


    

    


