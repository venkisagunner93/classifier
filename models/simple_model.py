import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        
        # Block 1: 224x224x3 → 112x112x16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.25)  # Dropout after Block 1
        
        # Block 2: 112x112x16 → 56x56x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.25)  # Dropout after Block 2
        
        # Block 3: 56x56x32 → 28x28x64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(0.25)  # Dropout after Block 3
        
        # MaxPool layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Final pooling to reduce size further: 28x28x64 → 7x7x64
        self.global_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layer for classification
        self.fc = nn.Linear(7 * 7 * 64, 2)  # Assuming binary classification (Apple/Banana)
        
    def forward(self, x):
        x = self.pool(self.dropout1(self.bn1(self.conv1(x))))
        x = self.pool(self.dropout2(self.bn2(self.conv2(x))))
        x = self.pool(self.dropout3(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc(x)
        return x
