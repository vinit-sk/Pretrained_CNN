#Dataset Loader

#import lib
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_train_val_dataset(data_root, 
                           batch_size=32,
                           train_transforms = None, 
                           val_test_transforms=None, 
                           ):
    if train_transforms is None: 
        train_transforms = train_transform
    if val_test_transforms is None:
        val_test_transforms=val_test_transform
    train_path = data_root + 'train'
    train_full = datasets.ImageFolder(train_path, transform=train_transforms)
    # Class train_transform
    class_names = train_full.classes    
    # Train/val split >> from training data
    val_size = int(0.1 * len(train_full))
    train_size = len(train_full) - val_size
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

    # Apply test/val transform to val set
    val_dataset.dataset.transform = val_test_transforms

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, class_names

        
def load_test_dataset(data_root, 
                      batch_size=32,
                      transform = None,  
                      ):
    test_path = data_root +'test'
    if transform is None:
        transform = val_test_transform
    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader