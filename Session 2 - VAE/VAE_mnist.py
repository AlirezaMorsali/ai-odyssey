import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def plot_images_grid(images, nrows, ncols, title=""):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].imshow(
                images[i * ncols + j].cpu().numpy().squeeze(), cmap="gray"
            )
            axes[i, j].axis("off")
    plt.suptitle(title)
    plt.pause(0.001)
    plt.close(fig)


class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # mu
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # sigma

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.ModuleDict):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=latent_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon


class VAE(nn.Module):
    def __init__(self, decoder, encoder) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _reparam(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + (eps * logvar)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self._reparam(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


def loss_function(x_recon, mu, logvar, data):
    recon_loss = nn.functional.binary_cross_entropy(x_recon, data, reduction="sum")
    # recon_loss = nn.functional.binary_cross_entropy(x_recon, data)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    beta = 0
    return recon_loss + (beta * kl_div)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 10
latent_dim = 20

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


sample_data, _ = next(iter(train_loader))
sample_data = sample_data.view(-1, 784)[:16].to(device)
plot_images_grid(
    sample_data.view(-1, 1, 28, 28),
    nrows=4,
    ncols=4,
    title="Original Images Before Training",
)


def generate_digit(vae, latent_dim=20):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)  # Sample random latent vectors

        generated_images = vae.decoder(z).view(-1, 1, 28, 28)
        plot_images_grid(
            generated_images,
            nrows=4,
            ncols=4,
            title=f"Generated Images of Digit",
        )


encoder = Encoder()
decoder = Decoder()

vae = VAE(encoder=encoder, decoder=decoder)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

step_count = 0
for epoch in range(num_epochs):

    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)
        optimizer.zero_grad()

        x_recon, mu, logvar = vae(data)
        loss = loss_function(x_recon, mu, logvar, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        step_count += 1

        if step_count % 50 == 0:
            with torch.no_grad():
                recon_images = vae(data[:16])[0]
                plot_images_grid(
                    recon_images.view(-1, 1, 28, 28),
                    nrows=4,
                    ncols=4,
                    title=f"Reconstructed Images at Step {step_count}",
                )

                generate_digit(vae, latent_dim=20)
    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}")
