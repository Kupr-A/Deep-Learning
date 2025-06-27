import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from CNN_struct import DualHeadCNN
from lab2_data_download import fashion_test
from torch.utils.data import DataLoader

device = torch.device('cpu')


model = DualHeadCNN().to(device)
checkpoint = torch.load("model_heads_fashion_finetuned.pth", map_location=device)
model.shared.load_state_dict(checkpoint['shared'])
model.head2.load_state_dict(checkpoint['head2'])
model.eval()


test_loader = DataLoader(fashion_test, batch_size=64, shuffle=False)


confusions = {
    (c, t): (-1, None) for c in range(10) for t in range(10) if c != t
}


with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images, head=2)
        probs = F.softmax(logits, dim=1)

        for i in range(len(images)):
            true_class = labels[i].item()
            for t in range(10):
                if t == true_class:
                    continue
                prob = probs[i, t].item()
                key = (true_class, t)
                if prob > confusions[key][0]:
                    confusions[key] = (prob, images[i].cpu())

fig, axes = plt.subplots(10, 9, figsize=(15, 18))
fig.suptitle("для пар c, t: изображение класса c похожее на t", fontsize=16)

for idx, ((c, t), (prob, image)) in enumerate(confusions.items()):
    row = c
    col = t if t < c else t - 1
    ax = axes[row, col]
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f"{c} → {t}\nP={prob:.2f}")
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig("most_confused_images.png")
plt.show()
