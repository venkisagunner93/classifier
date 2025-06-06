import os
import logging
from typing import List, Tuple, Set
from pathlib import Path
from PIL import Image, ImageOps
import json
import argparse

class Dataset:
    """Handles dataset operations for image classification."""
    
    VALID_IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    def __init__(self, data_folder: str = 'data'):
        """
        Initialize Dataset processor.
        
        Args:
            data_folder: Root directory for dataset
        """
        self.data_folder = Path(data_folder)
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging for the dataset operations."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_image_files(self, folder_path: Path) -> List[Path]:
        """
        Get all image files from the specified folder.
        
        Args:
            folder_path: Directory to search for images
            
        Returns:
            List of image file paths
        """
        return [
            f for f in folder_path.iterdir()
            if f.suffix.lower() in self.VALID_IMAGE_EXTENSIONS
        ]
    
    def rename_files_to_format(self, folder_path: str, prefix: str = "apple") -> None:
        """
        Rename all image files in the given folder to a standardized format.
        
        Args:
            folder_path: Directory containing images to rename
            prefix: Prefix to use for renamed files
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            self.logger.error(f"Folder '{folder}' does not exist.")
            return
        
        image_files = self._get_image_files(folder)
        image_files.sort()
        
        self.logger.info(f"Found {len(image_files)} image files to rename...")
        
        for i, file_path in enumerate(image_files, start=1):
            new_name = folder / f"{prefix}_{str(i).zfill(3)}.jpg"
            try:
                file_path.rename(new_name)
                self.logger.info(f"Renamed: {file_path.name} -> {new_name.name}")
            except OSError as e:
                self.logger.error(f"Error renaming {file_path.name}: {e}")

    def resize_images(self, folder_path, target_size=(224, 224), 
                      resize_method="crop", quality=95, 
                      normalize=True, create_backup=True):
        """Advanced image resizing for deep learning with multiple resize methods."""
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' does not exist.")
            return
        
        if create_backup:
            backup_folder = os.path.join(folder_path, "original_size_backup")
            os.makedirs(backup_folder, exist_ok=True)
            print(f"Backup folder created: {backup_folder}")
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
        
        print(f"Resizing {len(image_files)} images to {target_size} using '{resize_method}' method...")
        
        for filename in image_files:
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    if create_backup:
                        backup_path = os.path.join(backup_folder, filename)
                        img.save(backup_path)
                    
                    if normalize and img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    if resize_method == "stretch":
                        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    elif resize_method == "crop":
                        resized_img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
                    elif resize_method == "pad":
                        resized_img = ImageOps.pad(img, target_size, Image.Resampling.LANCZOS)
                    
                    base_name = os.path.splitext(filename)[0]
                    new_filename = f"{base_name}.jpg"
                    new_path = os.path.join(folder_path, new_filename)
                    resized_img.save(new_path, 'JPEG', quality=quality, optimize=True)
                    
                    if new_filename != filename:
                        os.remove(file_path)
                    
                    print(f"✓ {filename} -> {target_size}")
            except Exception as e:
                print(f"✗ Error with {filename}: {e}")
        
        print(f"\nResize complete! All images are now {target_size}")

    def rename_and_create_metadata(self, folder_path, output_json="metadata.json"):
        """
        Rename all images in the folder to image_000x.jpg, split the dataset into
        train/val/test sets, and create a JSON file mapping the new image paths
        to their original class, dimensions, and split type.

        Args:
            folder_path: Path to the folder containing images.
            output_json: Path to save the metadata JSON file.
        """
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' does not exist.")
            return

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
        files.sort()

        metadata = {}
        print(f"Found {len(files)} image files to rename")

        for filename in enumerate(files, start=1):
            old_path = os.path.join(folder_path, filename)
            new_name = f"image_{str(i).zfill(4)}.jpg"
            new_path = os.path.join(folder_path, new_name)

            # Extract class from the original filename (e.g., "apple" or "banana")
            image_class = "apple" if "apple" in filename.lower() else "banana"

            try:
                with Image.open(old_path) as img:
                    dimensions = img.size  # (width, height)
                    os.rename(old_path, new_path)
                    metadata[new_path] = {
                        "class": image_class,
                        "dimensions": {"width": dimensions[0], "height": dimensions[1]},
                    }
                    print(f"Renamed: {filename} -> {new_name}")
            except OSError as e:
                print(f"Error renaming {filename}: {e}")

        # Save metadata to JSON
        with open(output_json, "w") as json_file:
            json.dump(metadata, json_file, indent=4)

        print(f"Renaming and splitting complete! Metadata saved to {output_json}")
    
    def split_dataset(self, metadata, train_ratio=0.8, val_ratio=0.1):
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            metadata: Path to metadata.json file.
            train_ratio: Proportion of data to use for training.
            val_ratio: Proportion of data to use for validation.
        """
        # Open the metadata file
        with open(metadata, 'r') as f:
            data = json.load(f)

        # Extract entries and create dict against each class
        entries = list(data.items())
        class_dict = {}
        for path, info in entries:
            cls = info['class']
            if cls not in class_dict:
                class_dict[cls] = []
            class_dict[cls].append((path, info))
        
        # Iterate through class_dict to split data
        train_data = []
        val_data = []
        test_data = []
        for cls, items in class_dict.items():
            n = len(items)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_data.extend(items[:train_end])
            val_data.extend(items[train_end:val_end])
            test_data.extend(items[val_end:])
        
        # Print split statistics by class
        print("Dataset split statistics:")
        print(f"Total classes: {len(class_dict)}")
        for cls in class_dict.keys():
            total_count = len(class_dict[cls])
            train_count = sum(1 for path, info in train_data if info['class'] == cls)
            val_count = sum(1 for path, info in val_data if info['class'] == cls)
            test_count = sum(1 for path, info in test_data if info['class'] == cls)
            print(f"{cls}: Total={total_count}, Train={train_count}, Val={val_count}, Test={test_count}")

        # Now modify the metadata to include split information
        for path, info in train_data:
            info['split'] = 'train'
        for path, info in val_data:
            info['split'] = 'val'
        for path, info in test_data:
            info['split'] = 'test'

        # Save the modified metadata back to the file
        with open(metadata, 'w') as f:
            json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Process dataset images for classification')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Resize command
    resize_parser = subparsers.add_parser('resize', help='Resize images')
    resize_parser.add_argument('--input-dir', required=True, help='Input directory containing images')
    resize_parser.add_argument('--output-dir', required=True, help='Output directory for resized images')
    resize_parser.add_argument('--size', type=int, default=224, help='Size to resize images to (default: 224)')
    
    # Metadata command
    rename_parser = subparsers.add_parser('metadata', help='Create metadata and rename images')
    rename_parser.add_argument('--input-dir', required=True, help='Input directory containing images')
    rename_parser.add_argument('--output-json', required=True, help='Output JSON file for metadata')

    # Rename command
    rename_parser = subparsers.add_parser('rename', help='Rename images to standardized format')
    rename_parser.add_argument('--input-dir', required=True, help='Input directory containing images')
    rename_parser.add_argument('--prefix', default='apple', help='Prefix for renamed files (default: apple)')

    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test sets')
    split_parser.add_argument('--metadata', required=True, help='Path to metadata.json file')
    split_parser.add_argument('--train-ratio', type=float, default=0.8, help='Train set ratio (default: 0.8)')
    split_parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    dataset = Dataset()
    
    if args.command == 'resize':
        dataset.resize_images(args.input_dir, target_size=(args.size, args.size))
    elif args.command == 'metadata':
        dataset.rename_and_create_metadata(args.input_dir, args.output_json)
    elif args.command == 'rename':
        dataset.rename_files_to_format(args.input_dir, prefix=args.prefix)
    elif args.command == 'split':
        dataset.split_dataset(args.metadata, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
