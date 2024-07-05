import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import os

class ISARDataset(Dataset):
    """Partitions ISAR images into sequences by class
    """
    def __init__(self, img_dir, annotations_file, seq_length, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file, header=None, names=['image', 'class'])
        self.seq_length = seq_length
        self.transform = transform
        self.target_transform = target_transform
        self.class_groups = self.img_labels.groupby('class')

    def __len__(self):
        return sum(len(group) - self.seq_length + 1 for _, group in self.class_groups)

    def __getitem__(self, idx):
        # Find class index and index within the class
        for class_label, group in self.class_groups:
            group_size = len(group) - self.seq_length + 1
            if idx < group_size:
                group_idx = idx
                break
            else:
                idx -= group_size
        
        # Read sequence images
        img_paths = group.iloc[group_idx : group_idx + self.seq_length, 0].tolist()
        images = []
        for img_path in img_paths:
            image = read_image(os.path.join(self.img_dir, img_path)).float()
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images)    
        label = torch.tensor(class_label)

        if self.target_transform:
            label = self.target_transform(label)

        return images, label
