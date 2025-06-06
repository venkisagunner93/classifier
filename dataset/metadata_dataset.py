from torch.utils.data import Dataset
import json
from PIL import Image
from pathlib import Path

class MetadataDataset(Dataset):
    """Custom dataset that reads from metadata.json"""
    
    def __init__(self, metadata_path, split='train', transform=None):
        self.transform = transform
        self.split = split
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Filter data for the given split and extract training data
        self.data = []
        for path, info in metadata.items():
            if info['split'] == split:
                self.data.append((path, info['class']))
        
        # Create class to index mapping
        classes = sorted(set(info['class'] for info in metadata.values()))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.classes = classes
        
        print(f"Found {len(self.data)} images for {split} split")
        print(f"Classes: {self.classes}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label_str = self.data[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[label_str]
        return image, label