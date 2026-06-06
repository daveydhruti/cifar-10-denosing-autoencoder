"""
model.py
--------
Defines the convolutional denoising autoencoder architecture for CIFAR-10.

Architecture overview:
  Encoder: 3 strided Conv2d layers (3→64→128→256 channels) + a Linear bottleneck (512-d)
  Decoder: Linear projection back to spatial volume + 3 ConvTranspose2d layers (256→128→64→3)

The bottleneck compresses each 32×32×3 image (~3072 values) down to 512 dimensions,
forcing the network to learn compact, noise-invariant representations.
"""

import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    """
    Convolutional autoencoder that maps a noisy CIFAR-10 image to its clean version.

    Encoder path (spatial downsampling via stride-2 convolutions):
        Input  : (B, 3,   32, 32)
        After C1: (B, 64,  16, 16)
        After C2: (B, 128,  8,  8)
        After C3: (B, 256,  4,  4)
        Flatten: (B, 4096)
        Linear : (B, 512)   ← bottleneck

    Decoder path (spatial upsampling via transposed convolutions):
        Linear : (B, 4096)
        Unflatten: (B, 256, 4, 4)
        After T1: (B, 128,  8,  8)
        After T2: (B, 64,  16, 16)
        After T3: (B, 3,   32, 32)  ← Sigmoid keeps output in [0, 1]
    """

    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            # C1: downsample 32→16, expand channels 3→64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # C2: downsample 16→8, expand channels 64→128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # C3: downsample 8→4, expand channels 128→256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            # Bottleneck: 256*4*4=4096 → 512
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            # Project bottleneck back to spatial volume
            nn.Linear(512, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            # T1: upsample 4→8, compress channels 256→128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # T2: upsample 8→16, compress channels 128→64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # T3: upsample 16→32, compress channels 64→3 (RGB output)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Constrain output to [0, 1] to match normalised pixel range
        )

    def forward(self, x):
        """
        Args:
            x: Noisy input image tensor of shape (B, 3, 32, 32), values in [0, 1].
        Returns:
            Reconstructed clean image tensor of shape (B, 3, 32, 32), values in [0, 1].
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
