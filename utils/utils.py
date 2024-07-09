import os
import torch
from tqdm import tqdm
from utils.seed import seed_everything
from utils.dataloader import load_data, load_image


def train(model, dataloader, optimizer, criterion, device):
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in tqdm(dataloader, unit="batch", desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += len(labels)
    
    avg_loss = train_loss / len(dataloader)
    acc = train_correct / train_total

    return avg_loss, acc


def evaluate(model, dataloader, criterion, device):
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    for images, labels in tqdm(dataloader, unit="batch", desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        val_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        val_correct += (predictions == labels).sum().item()
        val_total += len(labels)

    avg_loss = val_loss / len(dataloader)
    acc = val_correct / val_total

    return avg_loss, acc


def test(model, device):
    _, _, dataloader = load_data(seq_length=5, batch_size=128)
    model.eval()

    test_correct = 0
    test_total = 0

    for images, labels in tqdm(dataloader, unit="batch", desc="Testing"):
        images.to(device)
        labels.to(device)

        with torch.no_grad():
            outputs = model(images)
        
        predictions = torch.argmax(outputs, dim=1)
        test_correct += (predictions == labels).sum().item()
        test_total += len(labels)

    return test_correct / test_total


def predict(model, img_path, device, label_map=None):
    model.eval()

    # Process path
    image_paths = []
    if os.path.isdir(img_path):
        for img in os.listdir(img_path):
            if img.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(img_path, img))
    else:
        image_paths.append(img_path)

    predictions = []
    for image_path in tqdm(image_paths, unit='img', desc='Predicting'):
        # Unsqueeze batch and sequence length dimensions
        image = load_image(image_path).unsqueeze(0).unsqueeze(0)
        image.to(device)

        with torch.no_grad():
            output = model(image)
        
        predicted_idx = torch.argmax(output, dim=1).item()
        predictions.append(label_map[predicted_idx] if label_map else predicted_idx)
    
    return [(image_path, prediction) for image_path, prediction in zip(image_paths, predictions)]


def fit(model, optimizer, criterion, num_epochs, device):
    seed_everything(0)

    train_loader, val_loader, _ = load_data(seq_length=5, batch_size=128)

    patience = 5
    best_val_loss = float('inf')
    epoch_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs}: loss: {train_loss}, acc: {train_acc}, val_loss: {val_loss}, val_acc: {val_acc}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epoch_counter = 0
        else:
            epoch_counter += 1
            if epoch_counter >= patience:
                print('Training stopped early!')
                return
