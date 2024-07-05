import torch
from tqdm import tqdm
from utils.seed import seed_everything
from utils.dataloader import load_data


def train(model, dataloader, optimizer, criterion, device):
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    # with tqdm(total=len(dataloader), unit="batch", desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}") as pbar:
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

        # # Update progress bar
        # pbar.update(1)
        # pbar.set_postfix(loss=train_loss / train_total, acc=train_correct / train_total)
    
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


def test(model, dataloader, device):
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

def fit(model, optimizer, criterion, num_epochs, device):
    seed_everything(0)

    train_loader, val_loader, _ = load_data(seq_length=3, batch_size=4)

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
