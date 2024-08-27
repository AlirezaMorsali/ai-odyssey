import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 10
latent_dim = 20
condition_dim = 10  # For digit labels (0-9)

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Helper function to plot a grid of images
def plot_images_grid(images, nrows, ncols, title=""):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].imshow(
                images[i * ncols + j].cpu().numpy().squeeze(), cmap="gray"
            )
            axes[i, j].axis("off")
    plt.suptitle(title)
    plt.pause(0.001)  # Non-blocking pause
    plt.close(fig)  # Close the figure to free up resources


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, condition_dim=10):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)  # Concatenate image data and label
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784, condition_dim=10):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, c):
        z = torch.cat([z, c], dim=1)  # Concatenate latent vector and label
        h = torch.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon


# VAE Model
class ConditionalVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(ConditionalVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar


# Loss function (Reconstruction + KL Divergence losses summed over all elements and batch)
def loss_function(x_recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


# Model, optimizer
encoder = Encoder().to(device)
decoder = Decoder().to(device)
cvae = ConditionalVAE(encoder, decoder).to(device)
optimizer = optim.Adam(cvae.parameters(), lr=learning_rate)


# One-hot encoding function for labels
def one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes).float()


# Training
cvae.train()
step_count = 0
for epoch in range(num_epochs):
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)
        labels = one_hot(labels).to(device)  # Convert labels to one-hot encoding

        optimizer.zero_grad()
        x_recon, mu, logvar = cvae(data, labels)
        loss = loss_function(x_recon, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        step_count += 1

        # Plot reconstructed images every 10 steps
        if step_count % 10 == 0:
            with torch.no_grad():
                recon_images = cvae(data[:16], labels[:16])[0]
                plot_images_grid(
                    recon_images.view(-1, 1, 28, 28),
                    nrows=4,
                    ncols=4,
                    title=f"Reconstructed Images at Step {step_count}",
                )

    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}")

# Plot reconstructed images after training
cvae.eval()
with torch.no_grad():
    final_recon_images = cvae(data[:16], labels[:16])[0]
    plot_images_grid(
        final_recon_images.view(-1, 1, 28, 28),
        nrows=4,
        ncols=4,
        title="Reconstructed Images After Training",
    )

# Save the trained model
torch.save(cvae.state_dict(), "conditional_vae_mnist.pth")


# Generate images for specific digits
def generate_digit(cvae, digit, latent_dim=20):
    cvae.eval()
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)  # Sample random latent vectors
        label = torch.tensor([digit] * 16).to(
            device
        )  # Create a batch of the same digit
        label_one_hot = one_hot(label).to(device)
        generated_images = cvae.decoder(z, label_one_hot).view(-1, 1, 28, 28)
        plot_images_grid(
            generated_images,
            nrows=4,
            ncols=4,
            title=f"Generated Images of Digit {digit}",
        )


# Generate specific digits
generate_digit(cvae, 0)  # Change the digit value to generate other digits
