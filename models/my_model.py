import torch
import torch.nn as nn

# YOUR original model architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # CNN Backbone - your design
        # Block 1: 224x224x3 → 112x112x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: 112x112x32 → 56x56x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: 56x56x64 → 28x28x128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4: 28x28x128 → 14x14x128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Block 5: 14x14x128 → 7x7x128
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # YOUR MLP Head: 7x7x128 = 6,272 → 512 → 128 → 2
        self.fc1 = nn.Linear(7 * 7 * 128, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # CNN Backbone
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        
        # Flatten for MLP
        x = torch.flatten(x, 1)
        
        # YOUR MLP Head
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation - use with CrossEntropyLoss
        
        return x
