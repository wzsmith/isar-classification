# Link to paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9971386
import torch
import torch.nn as nn
from torch.optim import Adam
from utils.seed import seed_everything
from utils.utils import fit
from model import CNN_BiLSTM

def main():
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN_BiLSTM(num_classes=4)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    fit(model, optimizer, criterion, num_epochs=20, device=device)

    torch.save(model.state_dict(), 'test1.pth')

if __name__ == '__main__':
    main()