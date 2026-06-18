from ultralytics import YOLO
from gradcam import GradCAM
import cv2
import torch
from torch import nn



img_path = "6_jpg.rf.b3dbdcd56ecce6980a163a59431dad5d.jpg"

yolo_model = YOLO("runs/detect/train/weights/best.pt")
results = yolo_model(img_path)
annotated_frame = results[0].plot()
cv2.imshow("YOLO", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

class CancerCNN(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
      self.relu1 =nn.ReLU()
      self.maxpool1 = nn.MaxPool2d(2)


 


#cnn_model = CancerCNN()
#cnn_model.eval()


    


    

    


