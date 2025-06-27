import torch
from torch import nn
import matplotlib.pyplot as plt

from CNN_struct import DualHeadCNN
from lab2_data_download import train_loader, test_loader

def train_model(model, dataloader, criterion, optimizer, device, head=1):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, head=head)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, device, head=1):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, head=head)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


device = torch.device('cpu')
model = DualHeadCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_losses, val_accuracies = [], []



for epoch in range(10):
    loss = train_model(model, train_loader, criterion, optimizer, device, head=1)
    acc = evaluate_model(model, test_loader, device, head=1)

    train_losses.append(loss)
    val_accuracies.append(acc)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Val Acc={acc:.4f}")



torch.save({
    'shared': model.shared.state_dict(),
    'head1': model.head1.state_dict(),
    'head2': model.head2.state_dict()
}, "model_heads_checkpoint.pth")



plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.legend()
plt.savefig("training_loss.png")


plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy vs Epochs")
plt.legend()
plt.savefig("validation_accuracy.png")
