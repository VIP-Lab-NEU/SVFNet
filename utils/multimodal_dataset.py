import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from utils.utils import cvtColor, preprocess_input


class MultiModalClassificationDataset(Dataset):
    """
    Multi-modal multi-label classification dataset
    Supports loading SONAR and RGB image pairs with multi-hot labels
    Includes single-modal mode to avoid loading unnecessary modality data
    """
    
    def __init__(self, scene_list_file, sonar_root, rgb_root, 
                 input_shape=[224, 224], num_classes=10, 
                 train=True, transform_sonar=None, transform_rgb=None,
                 baseline_mode='multimodal'):
        """
        Args:
            scene_list_file: Scene list file path (train_scenelist.txt or val_scenelist.txt)
            sonar_root: SONAR image root directory
            rgb_root: RGB image root directory
            input_shape: Input image size [H, W]
            num_classes: Total number of classes
            train: Whether in training mode
            transform_sonar: SONAR image transform
            transform_rgb: RGB image transform
            baseline_mode: Baseline mode ('multimodal', 'rgb_only', 'sonar_only')
        """
        self.sonar_root = sonar_root
        self.rgb_root = rgb_root
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.baseline_mode = baseline_mode
        
        self.samples = []
        with open(scene_list_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= num_classes + 1:
                image_name = parts[0]
                labels = [int(x) for x in parts[1:num_classes + 1]]
                self.samples.append((image_name, labels))
        
        if transform_sonar is None:
            if train:
                self.transform_sonar = transforms.Compose([
                    transforms.Resize((input_shape[0], input_shape[1])),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform_sonar = transforms.Compose([
                    transforms.Resize((input_shape[0], input_shape[1])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform_sonar = transform_sonar
            
        if transform_rgb is None:
            if train:
                self.transform_rgb = transforms.Compose([
                    transforms.Resize((input_shape[0], input_shape[1])),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform_rgb = transforms.Compose([
                    transforms.Resize((input_shape[0], input_shape[1])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform_rgb = transform_rgb
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Return a data sample with retry mechanism to skip corrupted data
        """
        while True:
            try:
                image_name, labels = self.samples[index]
                
                labels_tensor = torch.tensor(labels, dtype=torch.float32)
                
                if self.baseline_mode == 'rgb_only':
                    rgb_path = os.path.join(self.rgb_root, image_name)
                    rgb_image = Image.open(rgb_path).convert('RGB')
                    rgb_tensor = self.transform_rgb(rgb_image)
                    return rgb_tensor, labels_tensor
                    
                elif self.baseline_mode == 'sonar_only':
                    sonar_path = os.path.join(self.sonar_root, image_name)
                    sonar_image = Image.open(sonar_path).convert('RGB')
                    sonar_tensor = self.transform_sonar(sonar_image)
                    return sonar_tensor, labels_tensor
                    
                else:  # multimodal mode
                    sonar_path = os.path.join(self.sonar_root, image_name)
                    rgb_path = os.path.join(self.rgb_root, image_name)
                    
                    sonar_image = Image.open(sonar_path).convert('RGB')
                    rgb_image = Image.open(rgb_path).convert('RGB')
                    
                    sonar_tensor = self.transform_sonar(sonar_image)
                    rgb_tensor = self.transform_rgb(rgb_image)
                    
                    return sonar_tensor, rgb_tensor, labels_tensor

            except Exception as e:
                print(f"Warning: Failed to load index {index} ({self.samples[index][0]}), trying next... | Error: {e}")
                index = (index + 1) % len(self.samples)


def multimodal_collate_fn(batch):
    """
    Collate function for multi-modal data
    Supports None value handling in single-modal mode
    """
    sonar_images, rgb_images, labels = zip(*batch)
    
    if all(img is None for img in sonar_images):
        sonar_batch = None
        rgb_batch = torch.stack([img for img in rgb_images if img is not None], 0)
    elif all(img is None for img in rgb_images):
        sonar_batch = torch.stack([img for img in sonar_images if img is not None], 0)
        rgb_batch = None
    else:
        sonar_batch = torch.stack(sonar_images, 0)
        rgb_batch = torch.stack(rgb_images, 0)
    
    labels_batch = torch.stack(labels, 0)
    
    return sonar_batch, rgb_batch, labels_batch


def single_modal_collate_fn(batch):
    """
    Collate function for single-modal data
    Now __getitem__ only returns 2 values, making this function simpler
    """
    inputs, labels = zip(*batch)
    
    inputs_batch = torch.stack(inputs, 0)
    labels_batch = torch.stack(labels, 0)
    
    return inputs_batch, labels_batch


class MultiModalDataLoader:
    def __init__(self, dataset_dir, batch_size=32, input_shape=[224, 224], 
                 num_classes=10, num_workers=4, train=True):
        """
        Multi-modal data loader wrapper
        
        Args:
            dataset_dir: Dataset directory (e.g. dataset_classification/train)
            batch_size: Batch size
            input_shape: Input image size
            num_classes: Number of classes
            num_workers: Number of data loading threads
            train: Whether in training mode
        """
        if train:
            scene_list_file = os.path.join(dataset_dir, 'train_scenelist.txt')
        else:
            scene_list_file = os.path.join(dataset_dir, 'val_scenelist.txt')
            
        sonar_root = os.path.join(dataset_dir, 'soanr')
        rgb_root = os.path.join(dataset_dir, 'rgb')
        
        self.dataset = MultiModalClassificationDataset(
            scene_list_file=scene_list_file,
            sonar_root=sonar_root,
            rgb_root=rgb_root,
            input_shape=input_shape,
            num_classes=num_classes,
            train=train
        )
        
        from torch.utils.data import DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=multimodal_collate_fn,
            drop_last=train
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
