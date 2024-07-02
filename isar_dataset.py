import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import os

class ISARDataset(Dataset):
    def __init__(self, annotations_file, img_dir, seq_length=3, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.seq_length = seq_length
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels) - self.seq_length + 1

    def __getitem__(self, idx):
        # Read sequence images
        img_paths = self.img_labels.iloc[idx : idx + seq_length, 0].tolist()
        images = [read_image(os.path.join(self.img_dir, img_path)).float() for img_path in img_paths]
        images = torch.stack(images)    

        label = self.img_labels.iloc[idx, 1]
    
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label