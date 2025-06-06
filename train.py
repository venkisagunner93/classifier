import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
from dataset.metadata_dataset import MetadataDataset
import os

class TrainModel:
    def __init__(self, model_type="simple_model", data_folder="data", num_epochs=20, learning_rate=0.001, skip_training=False):
        self.__model_type = model_type
        self.__data_folder = data_folder
        self.__num_epochs = num_epochs
        self.__learning_rate = learning_rate
        self.__device = self._setup_device()
        self.__skip_training = skip_training
        
    def _setup_device(self):
        """Setup device for training"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            print("MPS not available, using CPU")
        return device
    
    def _get_model(self):
        """Get the specified model"""
        if self.__model_type == "simple_model":
            from models.simple_model import SimpleModel
            return SimpleModel()
        elif self.__model_type == "my_model":
            from models.my_model import MyModel
            return MyModel()
        else:
            raise ValueError(f"Unknown model type: {self.__model_type}")
    
    def _load_model(self, model):
        """Load model weights from disk - fixed for MPS compatibility"""
        model_path = f'{self.__model_type}_trained.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load on CPU first, then move to device (MPS compatibility fix)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.to(self.__device)  # Move to MPS after loading
        print(f"📥 Loaded model from '{model_path}'")
        return model

    
    def _get_train_transforms(self, no_random=False):
        """Get training transforms with data augmentation"""
        
        if no_random:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def create_loader(self, batch_size=None, split='train'):
        """Create training data loader from metadata.json"""
        # Auto-adjust batch size based on device
        if batch_size is None:
            batch_size = 8 if self.__device.type == "mps" else 16
        
        # Path to metadata file
        metadata_path = os.path.join(self.__data_folder, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Create training dataset from metadata
        dataset = MetadataDataset(
            metadata_path=metadata_path,
            split=split,
            transform=self._get_train_transforms(no_random=(split == 'test'))
        )
        
        # Create training data loader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Training samples: {len(dataset)}")
        print(f"Classes: {dataset.classes}")
        print(f"Batch size: {batch_size}")
        
        return loader
    
    def train_one_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.__device), labels.to(self.__device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        return epoch_loss, epoch_accuracy
    
    def validate(self, model, val_loader):
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.__device), labels.to(self.__device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct_val / total_val
        return val_accuracy
    
    def test(self, model, test_loader):
        # Testing phase
        model.eval()
        correct_test = 0
        total_test = 0
        class_correct = [0, 0]
        class_total = [0, 0]

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.__device), labels.to(self.__device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
                # Class-wise accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        test_accuracy = 100 * correct_test / total_test
        class_accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(len(class_correct))]

        return test_accuracy, class_accuracies
    
    def plot_training_progress(self, train_losses, train_accuracies):
        """Plot training progress"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title(f'Training Loss - {self.__model_type}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies)
        plt.title(f'Training Accuracy - {self.__model_type}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig(f'{self.__model_type}_training_progress.png')
        plt.show()
    
    def train(self, batch_size=None, save_model=True):
        """Main training function - only training, no validation/testing"""
        # Initialize model
        model = self._get_model().to(self.__device)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        if self.__skip_training:
            print("⏩ Skipping training, loading model from disk...")
            model = self._load_model(model)

            # Final evaluation on test set
            test_loader = self.create_loader(batch_size, split='test')
            test_accuracy, class_accuracies = self.test(model, test_loader)
            print(f"Test Accuracy: {test_accuracy:.2f}%")
            for i, acc in enumerate(class_accuracies):
                print(f"Class {i} Accuracy: {acc:.2f}%")
        else:
            print(f"🚀 Starting training with {self.__model_type}...")
        
            # Setup training data loader
            train_loader = self.create_loader(batch_size)
            val_loader = self.create_loader(batch_size, split='val')  # Using same loader for simplicity
            
            # Initialize model
            model = self._get_model().to(self.__device)
            print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.__learning_rate, weight_decay=1e-4)
            
            # Training loop
            train_losses = []
            train_accuracies = []
            
            start_time = time.time()
            
            for epoch in range(self.__num_epochs):
                # Train one epoch
                train_loss, train_acc = self.train_one_epoch(model, train_loader, criterion, optimizer)
                val_acc = self.validate(model, val_loader)
                
                # Store results
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                
                # Print progress
                print(f'Epoch [{epoch+1}/{self.__num_epochs}] - '
                    f'Loss: {train_loss:.4f}, '
                    f'Train Acc: {train_acc:.2f}%, '
                    f'Validation Acc: {val_acc:.2f}%')
            
            training_time = time.time() - start_time
            print(f"\n✅ Training completed in {training_time:.2f} seconds")

            # Save model
            if save_model:
                model_path = f'{self.__model_type}_trained.pth'
                torch.save(model.state_dict(), model_path)
                print(f"\n💾 Model saved as '{model_path}'")

            # Plot training progress
            self.plot_training_progress(train_losses, train_accuracies)
        
        return model

# Usage examples:
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a deep learning model.")
    parser.add_argument("--model_type", type=str, default="simple_model", choices=["simple_model", "my_model"],
                        help="Type of model to train (default: simple_model)")
    parser.add_argument("--data_folder", type=str, default="data",
                        help="Path to the data folder containing metadata.json (default: data)")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate for the optimizer (default: 0.001)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training (default: auto-adjust based on device)")
    parser.add_argument("--save_model", action="store_true",
                        help="Flag to save the trained model (default: False)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and load existing model from disk (default: False)")

    args = parser.parse_args()

    trainer = TrainModel(
        model_type=args.model_type,
        data_folder=args.data_folder,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        skip_training=args.skip_training
    )
    trainer.train(batch_size=args.batch_size, save_model=args.save_model)
