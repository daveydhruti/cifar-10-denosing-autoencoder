"""
data.py
-------
Data loading utilities for the CIFAR-10 denoising autoencoder experiment.

Two paired DataLoaders are returned for each split (train / test):
  - original_loader : clean images, no shuffle (preserves pairing with noisy loader)
  - noisy_loader    : same images with additive Gaussian noise applied on-the-fly

Because both loaders are built from deterministic transforms (the random seed is
fixed before this module is called in train.py), corresponding indices always
yield a matched (noisy, clean) pair even when iterating in parallel with zip().
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def make_transforms(noise_scale: float = 0.2):
    """
    Build the clean and noisy transform pipelines.

    Args:
        noise_scale: Standard deviation of the Gaussian noise added to each pixel.
                     Must be in [0.2, 0.5] as per the original assignment spec.
    Returns:
        transform_original: ToTensor only — produces clean images in [0, 1].
        transform_noisy   : ToTensor + additive Gaussian noise, clipped to [0, 1].
    """
    assert 0.2 <= noise_scale <= 0.5, "noise_scale must be between 0.2 and 0.5"

    transform_original = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_noisy = transforms.Compose([
        transforms.ToTensor(),
        # Add zero-mean Gaussian noise and clip back to valid pixel range
        transforms.Lambda(
            lambda x: torch.clamp(x + torch.randn(x.size()) * noise_scale, 0.0, 1.0)
        ),
    ])

    return transform_original, transform_noisy


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def load_cifar10(
    noise_scale: float = 0.2,
    batch_size: int = 64,
    data_root: str = "./data",
):
    """
    Download (if needed) and load CIFAR-10, returning four DataLoaders.

    Shuffle is intentionally disabled on all loaders so that parallel
    iteration with zip() always yields correctly matched (noisy, clean) pairs.

    Args:
        noise_scale : Gaussian noise standard deviation (see make_transforms).
        batch_size  : Mini-batch size used for all loaders.
        data_root   : Directory where the CIFAR-10 dataset is cached.

    Returns:
        original_trainloader : Clean training images.
        original_testloader  : Clean test images.
        noisy_trainloader    : Noisy training images (paired with original_trainloader).
        noisy_testloader     : Noisy test images (paired with original_testloader).
    """
    transform_original, transform_noisy = make_transforms(noise_scale)

    loader_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=2)

    original_trainloader = DataLoader(
        CIFAR10(root=data_root, train=True,  download=True, transform=transform_original),
        **loader_kwargs,
    )
    original_testloader = DataLoader(
        CIFAR10(root=data_root, train=False, download=True, transform=transform_original),
        **loader_kwargs,
    )
    noisy_trainloader = DataLoader(
        CIFAR10(root=data_root, train=True,  download=True, transform=transform_noisy),
        **loader_kwargs,
    )
    noisy_testloader = DataLoader(
        CIFAR10(root=data_root, train=False, download=True, transform=transform_noisy),
        **loader_kwargs,
    )

    return original_trainloader, original_testloader, noisy_trainloader, noisy_testloader
