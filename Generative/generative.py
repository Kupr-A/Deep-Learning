import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE

BATCH_SIZE = 128
LATENT_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda")


def show_images(images, title=""):
    images = images[:10].detach().cpu().view(-1, 1, 28, 28)
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(images[i][0], cmap="gray")
        axes[i].axis("off")
    plt.suptitle(title)
    plt.show()


def plot_training_loss(loss_values):
    plt.plot(loss_values)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


class EncoderDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train_model(model, loader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_losses = []

    for _ in range(epochs):
        total_loss = 0
        for batch_images, _ in loader:
            batch_images = batch_images.to(DEVICE)
            reconstructed = model(batch_images)
            loss = criterion(reconstructed, batch_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_losses.append(total_loss / len(loader))
        print(f"Loss: {epoch_losses[-1]:.4f}")

    return epoch_losses


def generate_gmm_samples(model, loader, sample_count=10):
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for images, _ in loader:
            z = model.encoder(images.to(DEVICE))
            latent_vectors.append(z.cpu())

    z_all = torch.cat(latent_vectors).numpy()
    gmm = GaussianMixture(n_components=10)
    gmm.fit(z_all)
    sampled_z, _ = gmm.sample(sample_count)
    decoded = model.decoder(torch.tensor(sampled_z).float().to(DEVICE))
    show_images(decoded, "Generated from GMM")


def generate_smote_samples(model, loader, dataset, sample_count=10):
    model.eval()
    latent_vectors = []
    targets = dataset.targets.numpy()

    with torch.no_grad():
        for images, _ in loader:
            z = model.encoder(images.to(DEVICE))
            latent_vectors.append(z.cpu())

    z_all = torch.cat(latent_vectors)
    sm = SMOTE()
    z_resampled, _ = sm.fit_resample(z_all.numpy(), targets)
    sampled_z = torch.tensor(z_resampled[:sample_count]).float().to(DEVICE)
    decoded = model.decoder(sampled_z)
    show_images(decoded, "Generated from SMOTE")


def main():
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EncoderDecoder(latent_dim=LATENT_SIZE).to(DEVICE)

    loss_history = train_model(model, loader, epochs=EPOCHS, lr=1e-3)
    plot_training_loss(loss_history)

    samples, _ = next(iter(loader))
    samples = samples.to(DEVICE)
    with torch.no_grad():
        reconstructions = model(samples)

    show_images(samples, "Original")
    show_images(reconstructions, "Reconstructed")

    generate_gmm_samples(model, loader)
    generate_smote_samples(model, loader, dataset)


if __name__ == "__main__":
    main()
