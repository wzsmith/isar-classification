from torch.utils.data import DataLoader, random_split
from torchvision.io import read_image
from torchvision import transforms
from utils.isar_dataset import ISARDataset

TRANSFORM = transforms.CenterCrop((120, 120))

def load_data(seq_length, batch_size):
    dataset = ISARDataset('data/', 'labels.csv', seq_length=seq_length, transform=TRANSFORM)

    train_ratio = 0.8
    val_ratio = 0.10

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_image(image_path):
    image = read_image(image_path).float()
    image = TRANSFORM(image)

    return image