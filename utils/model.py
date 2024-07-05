import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            ConvBlock(in_channels, 8),
            ConvBlock(8, 8),
            ConvBlock(8, 8),
            ConvBlock(8, 8),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(8, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(16, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        return x
    
class BiLSTM(nn.Module):
    def __init__(self, hidden_size, num_classes, num_layers=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(2592, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states (only needs to be done once)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))  

        # Extract last hidden state
        return out[:, -1, :]

class CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=1000):
        super().__init__()

        self.cnn = CNN()
        self.bilstm = BiLSTM(hidden_size, num_classes)
        self.fc1 = nn.Linear(2 * hidden_size, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        B, seq_length, C, H, W = x.shape

        # Process images in sequence via CNN independently
        cnn_out = []
        for i in range(seq_length):
            img = x[:, i, :, :, :]
            img = self.cnn(img)
            img = img.flatten(1)
            cnn_out.append(img)
        x = torch.stack(cnn_out, dim=1)
        
        x = self.bilstm(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x