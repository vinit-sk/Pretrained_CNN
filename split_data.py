import os
import shutil
import random
from tqdm import tqdm

def split_dataset_multi_class(source_dir, dest_dir, split_ratio=0.8):
    class_folders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in class_folders:
        src_class_path = os.path.join(source_dir, class_name)

        # List and shuffle images
        images = [f for f in os.listdir(src_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # Create train/test class folders
        train_class_path = os.path.join(dest_dir, 'train', class_name)
        test_class_path = os.path.join(dest_dir, 'test', class_name)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        print(f"\nProcessing class: {class_name}")
        
        # Copy train images with progress bar
        for img in tqdm(train_images, desc=f"Copying train images ({class_name})", leave=False):
            shutil.copy(os.path.join(src_class_path, img), os.path.join(train_class_path, img))

        # Copy test images with progress bar
        for img in tqdm(test_images, desc=f"Copying test images ({class_name})", leave=False):
            shutil.copy(os.path.join(src_class_path, img), os.path.join(test_class_path, img))

        print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")

# Example usage
source_dataset = '/root/VINIT/PetImages/'           # Folder with class subfolders
destination_dataset = '/root/VINIT/Dataset1/'        # Folder to save train/test splits
split_dataset_multi_class(source_dataset, destination_dataset, split_ratio=0.8)
