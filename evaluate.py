"""
evaluate.py
-----------
Evaluation and visualisation utilities for the trained denoising autoencoder.

Key functions:
  - evaluate()          : Compute per-image MSE on the test set.
  - show_worst()        : Display the N images with the largest reconstruction error.
  - show_noisy_pairs()  : Display N side-by-side original / noisy image pairs.
  - hyperparameter_study(): Sweep learning rates and batch sizes, plot results.

Usage (standalone):
    python evaluate.py --weights autoencoder.pth
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import load_cifar10
from model import DenoisingAutoencoder

SEED = 240167723
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    noisy_testloader,
    original_testloader,
    device: torch.device,
) -> dict:
    """
    Run the model over the full test set and collect per-image reconstruction errors.

    Args:
        model               : Trained DenoisingAutoencoder.
        noisy_testloader    : DataLoader of noisy test images.
        original_testloader : DataLoader of clean test images.
        device              : Computation device.

    Returns:
        Dictionary with keys:
          'errors'       – np.ndarray of per-image MSE values (shape: [N])
          'noisy'        – Tensor of all noisy test images    (shape: [N, 3, 32, 32])
          'original'     – Tensor of all clean test images    (shape: [N, 3, 32, 32])
          'reconstructed'– Tensor of all model outputs        (shape: [N, 3, 32, 32])
    """
    model.eval()
    errors, noisy_list, original_list, reconstructed_list = [], [], [], []

    with torch.no_grad():
        for (noisy, _), (original, _) in zip(noisy_testloader, original_testloader):
            noisy    = noisy.to(device)
            original = original.to(device)

            reconstructed = model(noisy)

            # Per-image MSE: average over channel, height, width dimensions
            per_image_mse = torch.mean((reconstructed - original) ** 2, dim=[1, 2, 3])
            errors.extend(per_image_mse.cpu().numpy())

            noisy_list.append(noisy.cpu())
            original_list.append(original.cpu())
            reconstructed_list.append(reconstructed.cpu())

    return {
        "errors":        np.array(errors),
        "noisy":         torch.cat(noisy_list,        dim=0),
        "original":      torch.cat(original_list,     dim=0),
        "reconstructed": torch.cat(reconstructed_list, dim=0),
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _to_hwc(tensor_chw):
    """Convert a (C, H, W) tensor to a clipped (H, W, C) numpy array for imshow."""
    return np.clip(tensor_chw.permute(1, 2, 0).numpy(), 0, 1)


def show_noisy_pairs(original_loader, noisy_loader, num_pairs: int = 10):
    """
    Display side-by-side original and noisy image pairs from the first batch.

    Args:
        original_loader : DataLoader of clean images.
        noisy_loader    : DataLoader of noisy images.
        num_pairs       : Number of pairs to display.
    """
    originals, _ = next(iter(original_loader))
    noisys, _    = next(iter(noisy_loader))

    fig, axes = plt.subplots(num_pairs, 2, figsize=(5, 2 * num_pairs))
    fig.suptitle("Original vs. Noisy Images", fontsize=13, y=1.01)

    for i in range(num_pairs):
        axes[i, 0].imshow(_to_hwc(originals[i]))
        axes[i, 0].set_title("Original", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(_to_hwc(noisys[i]))
        axes[i, 1].set_title("Noisy", fontsize=8)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("noisy_pairs.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: noisy_pairs.png")


def show_worst(results: dict, num_worst: int = 20):
    """
    Plot the N images with the highest reconstruction error as triples:
    (Original | Noisy | Reconstructed).

    Args:
        results  : Output dict from evaluate().
        num_worst: Number of worst-performing images to display.
    """
    worst_indices = np.argsort(-results["errors"])[:num_worst]

    fig, axes = plt.subplots(num_worst, 3, figsize=(8, 2.5 * num_worst))
    fig.suptitle(f"Worst {num_worst} Reconstructions (highest MSE)", fontsize=13, y=1.005)

    col_titles = ["Original", "Noisy", "Reconstructed"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9, fontweight="bold")

    for row, idx in enumerate(worst_indices):
        mse = results["errors"][idx]
        axes[row, 0].imshow(_to_hwc(results["original"][idx]))
        axes[row, 1].imshow(_to_hwc(results["noisy"][idx]))
        axes[row, 2].imshow(_to_hwc(results["reconstructed"][idx]))
        axes[row, 0].set_ylabel(f"MSE={mse:.4f}", fontsize=7, rotation=0, labelpad=55)

        for j in range(3):
            axes[row, j].axis("off")

    plt.tight_layout()
    plt.savefig("worst_reconstructions.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: worst_reconstructions.png")


# ---------------------------------------------------------------------------
# Hyperparameter study
# ---------------------------------------------------------------------------

def hyperparameter_study(
    num_epochs: int = 10,
    noise_scale: float = 0.2,
    device: torch.device = None,
):
    """
    Sweep learning rates and batch sizes independently; plot reconstruction error vs each.

    Learning rates tested : [0.0001, 0.001, 0.005, 0.01, 0.05]
    Batch sizes tested    : [16, 32, 64, 128, 256]

    For each sweep, all other hyperparameters are held at their default values.
    The mean test MSE is recorded for each configuration.

    Discussion of expected behaviour:
      - Very low LRs (e.g. 0.0001) converge too slowly within 10 epochs, yielding
        higher error.  Very high LRs (e.g. 0.05) can cause the loss to diverge or
        oscillate, also giving higher error.  The sweet spot lies in between.
      - Very small batch sizes (e.g. 16) introduce noisy gradient estimates that can
        act as implicit regularisation, sometimes helping, but also slow wall-clock
        training.  Very large batches (e.g. 256) can converge to sharper minima with
        worse generalisation (the "large-batch training" effect described by Keskar
        et al., 2017).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from train import train  # local import to avoid circular deps at module level

    # --- Learning rate sweep (batch size fixed at 64) ---
    learning_rates = [0.0001, 0.001, 0.005, 0.01, 0.05]
    lr_errors = []

    print("=== Learning Rate Sweep ===")
    for lr in learning_rates:
        torch.manual_seed(SEED)
        original_trainloader, original_testloader, noisy_trainloader, noisy_testloader = \
            load_cifar10(noise_scale=noise_scale, batch_size=64)

        model = DenoisingAutoencoder()
        train(
            model=model,
            noisy_trainloader=noisy_trainloader,
            original_trainloader=original_trainloader,
            num_epochs=num_epochs,
            lr=lr,                 # ← vary the learning rate here
            device=device,
            save_path=f"autoencoder_lr{lr}.pth",
        )
        results = evaluate(model, noisy_testloader, original_testloader, device)
        mean_mse = float(np.mean(results["errors"]))
        lr_errors.append(mean_mse)
        print(f"  LR={lr:.4f}  →  mean test MSE = {mean_mse:.6f}")

    # --- Batch size sweep (learning rate fixed at 0.001) ---
    batch_sizes = [16, 32, 64, 128, 256]
    bs_errors = []

    print("\n=== Batch Size Sweep ===")
    for bs in batch_sizes:
        torch.manual_seed(SEED)
        original_trainloader, original_testloader, noisy_trainloader, noisy_testloader = \
            load_cifar10(noise_scale=noise_scale, batch_size=bs)

        model = DenoisingAutoencoder()
        train(
            model=model,
            noisy_trainloader=noisy_trainloader,
            original_trainloader=original_trainloader,
            num_epochs=num_epochs,
            lr=0.001,
            device=device,
            save_path=f"autoencoder_bs{bs}.pth",
        )
        results = evaluate(model, noisy_testloader, original_testloader, device)
        mean_mse = float(np.mean(results["errors"]))
        bs_errors.append(mean_mse)
        print(f"  Batch size={bs:>3}  →  mean test MSE = {mean_mse:.6f}")

    # --- Plot both sweeps ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(learning_rates, lr_errors, marker="o", color="steelblue", linewidth=2)
    ax1.set_xscale("log")
    ax1.set_xlabel("Learning Rate (log scale)", fontsize=11)
    ax1.set_ylabel("Mean Test MSE", fontsize=11)
    ax1.set_title("Reconstruction Error vs. Learning Rate\n(batch size = 64, epochs = 10)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(batch_sizes, bs_errors, marker="s", color="darkorange", linewidth=2)
    ax2.set_xlabel("Batch Size", fontsize=11)
    ax2.set_ylabel("Mean Test MSE", fontsize=11)
    ax2.set_title("Reconstruction Error vs. Batch Size\n(lr = 0.001, epochs = 10)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("hyperparameter_study.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: hyperparameter_study.png")

    return {
        "learning_rates": learning_rates, "lr_errors": lr_errors,
        "batch_sizes": batch_sizes,       "bs_errors":  bs_errors,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the CIFAR-10 denoising autoencoder")
    parser.add_argument("--weights",     type=str,   default="autoencoder.pth", help="Path to saved model weights")
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--noise-scale", type=float, default=0.2)
    parser.add_argument("--num-worst",   type=int,   default=20, help="Number of worst images to visualise")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, original_testloader, _, noisy_testloader = load_cifar10(
        noise_scale=args.noise_scale,
        batch_size=args.batch_size,
    )

    model = DenoisingAutoencoder()
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    print(f"Loaded weights from '{args.weights}'")

    results = evaluate(model, noisy_testloader, original_testloader, device)
    print(f"\nMean test MSE: {np.mean(results['errors']):.6f}")

    show_worst(results, num_worst=args.num_worst)
