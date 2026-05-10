import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from PIL import Image

class ATTFacesDataset(Dataset):
    def __init__(self, train=True, transform=None):
        from sklearn.datasets import fetch_olivetti_faces
        # fetch_olivetti_faces returns 400 images of size 64x64, pixel values in [0, 1]
        data = fetch_olivetti_faces()
        images = data.images 
        targets = data.target 
        
        # Split 8 train / 2 test per subject
        train_idx = []
        test_idx = []
        for i in range(40):
            subject_indices = list(range(i * 10, (i + 1) * 10))
            train_idx.extend(subject_indices[:8])
            test_idx.extend(subject_indices[8:])
            
        indices = train_idx if train else test_idx
        
        self.images = images[indices]
        self.targets = targets[indices]
        self.transform = transform
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        # Convert [0, 1] float array to [0, 255] uint8 PIL Image
        img_np = (self.images[idx] * 255).astype(np.uint8)
        img = Image.fromarray(img_np, mode='L')
        label = int(self.targets[idx])
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_base_transform():
    """
    Base transformation for ResNet18: Resize to 224x224, convert to Tensor, and Normalize.
    Grayscale applied because both MNIST and AT&T are 1-channel, but ResNet expects 3.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_dataloaders(dataset_name='MNIST', batch_size=32, subset_size=None, test_transform=None):
    train_transform = get_base_transform()
    if test_transform is None:
        test_transform = get_base_transform()
        
    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        
        if subset_size is not None:
            # Use a small subset for rapid testing (e.g., 80/20 ratio maintained roughly)
            np.random.seed(42)
            train_indices = np.random.choice(len(train_dataset), size=subset_size, replace=False)
            test_indices = np.random.choice(len(test_dataset), size=min(subset_size // 5, len(test_dataset)), replace=False)
            train_dataset = Subset(train_dataset, train_indices)
            test_dataset = Subset(test_dataset, test_indices)
            
    elif dataset_name == 'AT&T':
        train_dataset = ATTFacesDataset(train=True, transform=train_transform)
        test_dataset = ATTFacesDataset(train=False, transform=test_transform)
        # We don't apply subset_size to AT&T since it's already very small (400 images total)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader
