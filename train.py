"""
train.py
--------
Training script for the CIFAR-10 denoising autoencoder.

Usage:
    python train.py                          # default settings
    python train.py --lr 0.001 --epochs 15  # custom hyperparameters

The trained model weights are saved to `autoencoder.pth` on completion.
"""

import argparse

import torch
import torch.nn as nn

from data import load_cifar10
from model import DenoisingAutoencoder

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 240167723
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    noisy_trainloader,
    original_trainloader,
    num_epochs: int = 10,
    lr: float = 0.001,
    device: torch.device = None,
    save_path: str = "autoencoder.pth",
) -> list:
    """
    Train the autoencoder by minimising MSE between reconstructed and clean images.

    Args:
        model               : Uninitialised DenoisingAutoencoder instance.
        noisy_trainloader   : DataLoader yielding noisy (corrupted) input images.
        original_trainloader: DataLoader yielding the corresponding clean images.
        num_epochs          : Number of full passes over the training set.
        lr                  : Adam optimiser learning rate.
        device              : Target device (CPU/CUDA). Auto-detected if None.
        save_path           : File path to save the trained model weights.

    Returns:
        epoch_losses: List of mean per-epoch training losses (one float per epoch).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for (noisy_images, _), (original_images, _) in zip(noisy_trainloader, original_trainloader):
            noisy_images    = noisy_images.to(device)
            original_images = original_images.to(device)

            # Forward pass: reconstruct clean image from noisy input
            reconstructed = model(noisy_images)
            loss = loss_fn(reconstructed, original_images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(noisy_trainloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1:>2}/{num_epochs}]  Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to '{save_path}'")
    return epoch_losses


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 denoising autoencoder")
    parser.add_argument("--epochs",      type=int,   default=10,    help="Number of training epochs")
    parser.add_argument("--lr",          type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--batch-size",  type=int,   default=64,    help="Mini-batch size")
    parser.add_argument("--noise-scale", type=float, default=0.2,   help="Gaussian noise std dev (0.2–0.5)")
    parser.add_argument("--save-path",   type=str,   default="autoencoder.pth", help="Output weights file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    original_trainloader, _, noisy_trainloader, _ = load_cifar10(
        noise_scale=args.noise_scale,
        batch_size=args.batch_size,
    )

    model = DenoisingAutoencoder()
    print(f"\nTraining  |  epochs={args.epochs}  lr={args.lr}  batch_size={args.batch_size}  noise_scale={args.noise_scale}\n")

    train(
        model=model,
        noisy_trainloader=noisy_trainloader,
        original_trainloader=original_trainloader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=args.save_path,
    )
