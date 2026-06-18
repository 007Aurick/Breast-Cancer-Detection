import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((64, 64)), #resizes image to 64x64
    transforms.ToTensor(),#converts to PyTorch tensor
])

dataset = ImageFolder(root="crops/", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


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

model = CancerCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for images, labels in dataloader:
        prediction = model(images)
        loss = criterion(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "cancer_cnn.pth")