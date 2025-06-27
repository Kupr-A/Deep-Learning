import torch
from matplotlib import pyplot as plt
from torch import nn
from CNN_struct import DualHeadCNN
from lab2_data_download import fashion_train, fashion_test
from torch.utils.data import DataLoader


device = torch.device('cpu')
model = DualHeadCNN().to(device)


checkpoint = torch.load("model_heads_checkpoint.pth", map_location=device)
model.shared.load_state_dict(checkpoint['shared'])
model.head2.load_state_dict(checkpoint['head2'])


for param in model.shared.parameters():
    param.requires_grad = False


optimizer = torch.optim.Adam(model.head2.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(fashion_train, batch_size=64, shuffle=True)
test_loader = DataLoader(fashion_test, batch_size=64)

train_losses, val_accuracies = [], []


for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, head=2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)


    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images, head=2).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    val_accuracies.append(acc)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={acc:.4f}")

torch.save({
    'shared': model.shared.state_dict(),
    'head1': model.head1.state_dict(),
    'head2': model.head2.state_dict()
}, "model_heads_fashion_trained.pth")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_accuracies, label="Val Accuracy")
plt.title("Frozen Conv Layers: Head 2")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("frozen_head2_curves.png")
plt.close()
