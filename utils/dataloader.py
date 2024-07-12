import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
from utils.isar_dataset import ISARDataset
from utils.constants import DATA_PATH, LABEL_PATH

TRANSFORM = transforms.CenterCrop((120, 120))

def load_data(seq_length, batch_size):
    dataset = ISARDataset(DATA_PATH, LABEL_PATH, seq_length=seq_length, transform=TRANSFORM)

    train_ratio = 0.8
    val_ratio = 0.10

    # Calculate indices for data splits
    train_indices, val_indices, test_indices = [], [], []
    start_idx = 0
    for i, count in dataset.counts.items():
        train_idx = start_idx + int(train_ratio * count)
        val_idx = start_idx + int((train_ratio + val_ratio) * count)

        train_indices.extend(range(start_idx, train_idx - seq_length)) # Account for sequence going past split
        val_indices.extend(range(train_idx, val_idx - seq_length))
        test_indices.extend(range(val_idx, start_idx + count))
        
        start_idx += count
    
    # Create Subset objects for each split
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def load_image(image_path):
    image = read_image(image_path).float()
    image = TRANSFORM(image)

    return image