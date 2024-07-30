import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm

torch.manual_seed(0)  # Set for testing purposes, please do not change!

batch_size = 128


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


dataloader = DataLoader(
    MNIST(".", download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
)
