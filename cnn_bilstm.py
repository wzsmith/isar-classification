# Link to paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9971386
import os
import argparse 

import torch
import torch.nn as nn
from torch.optim import Adam
from utils.seed import seed_everything
from utils.utils import test, fit
from utils.model import CNN_BiLSTM

seed_everything(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_BiLSTM(num_classes=4)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='')
    parser.add_argument('--test', action='store_true')
    # parser.add_argument('--seed', action='store_true')
    parser.add_argument('filepath', help='name of file to use for model')

    args = parser.parse_args()

    # Ensure file directory exists
    directory, filename = os.path.split(args.filepath)
    
    if not filename:
        parser.error('Please specify a filename.')
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created directory: {directory}')

    # Append .pth to model path
    if not (filename.endswith('.pt') or filename.endswith('.pth')):
        filename += '.pth'
    model_path = os.path.join(directory, filename)

    
    # Train model
    if args.train:
        print('Training model!')
        fit(model, optimizer, criterion, num_epochs=1, device=device)
        torch.save(model.state_dict(), model_path)
        print(f'Model saved at {model_path}')
    
    # Test model
    if args.test:
        # If only testing model
        if not args.train:
            print(f'Loading model from {model_path}')
            model.load_state_dict(torch.load(model_path))
        test_acc = test(model, device)
        print(f'Test accuracy: {test_acc}')

if __name__ == '__main__':
    main()