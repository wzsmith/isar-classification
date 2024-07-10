import os
import argparse 
import torch
import torch.nn as nn
from torch.optim import Adam
from utils.seed import seed_everything
from utils.utils import test, fit, predict
from utils.model import CNN_BiLSTM

seed_everything(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test model on test dataset')
    parser.add_argument('--predict', type=str, help='Path to a single image to predict') 
    parser.add_argument('filepath', help='Path to model file')

    args = parser.parse_args()

    # Initialize model
    model = CNN_BiLSTM(num_classes=4)
    model.to(device)

    # Ensure model path exists
    directory, filename = os.path.split(args.filepath)
    if not filename:
        parser.error('Please specify a filename.')
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created directory: {directory}')

    # Append .pth to model path
    if not (filename.endswith('.pt') or filename.endswith('.pth')):
        filename += '.pth'
    model_path = os.path.join(directory, filename)
    
    # Train model
    if args.train:
        optimizer = Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        print('Training model!')
        fit(model, optimizer, criterion, num_epochs=200, device=device)
        torch.save(model.state_dict(), model_path)
        print(f'Model saved at {model_path}')
    
    # Load model from file
    if args.test or args.predict and not args.train:
        model.load_state_dict(torch.load(model_path))
        print(f'Loaded model from {model_path}')

    # Test model against test dataset
    if args.test:
        test_acc = test(model, device)
        print(f'Test accuracy: {test_acc}')
    
    # Predict image or directory
    if args.predict:
        image_path = args.predict

        if not os.path.isfile(image_path) or os.path.isdir(image_path):
            raise FileNotFoundError(f'Please enter a valid image directory.')
        
        label_map = {
            0: 'Satellite',
            1: 'Asteroid', 
            2: 'Flat platform',
            3: 'Circular platform'
        }

        predictions = predict(model, image_path, device, label_map)
        for img_path, prediction in predictions:
            print(f'Predicted label for {img_path}: {prediction}')


if __name__ == '__main__':
    main()