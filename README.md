# Denoising Autoencoder — CIFAR-10

Convolutional autoencoder trained to remove Gaussian noise from 32×32 RGB images.

## Architecture

Encoder → 512-d bottleneck → Decoder, using strided convolutions throughout.

```
Input  (3, 32, 32)
  Conv2d(3→64,    k=4, s=2)  → (64,  16, 16)
  Conv2d(64→128,  k=4, s=2)  → (128,  8,  8)
  Conv2d(128→256, k=4, s=2)  → (256,  4,  4)
  Flatten → Linear(4096→512)     ← bottleneck
  Linear(512→4096) → Unflatten
  ConvTranspose2d(256→128)   → (128,  8,  8)
  ConvTranspose2d(128→64)    → (64,  16, 16)
  ConvTranspose2d(64→3)      → (3,   32, 32)
Output (3, 32, 32)  [Sigmoid]
```

- **Loss:** MSE against clean target
- **Optimiser:** Adam
- **Noise:** Additive Gaussian, σ = 0.2, clipped to [0, 1]

## Files

| File | Purpose |
|------|---------|
| `model.py` | Autoencoder definition |
| `data.py` | CIFAR-10 loaders with on-the-fly noise |
| `train.py` | Training loop + CLI |
| `evaluate.py` | Evaluation, visualisation, hyperparameter sweep |
| `notebook.ipynb` | End-to-end walkthrough |

## Usage

**Train**
```bash
python train.py                            # defaults: 10 epochs, lr=0.001, batch=64
python train.py --epochs 20 --lr 0.0005   # custom
```

**Evaluate**
```bash
python evaluate.py --weights autoencoder.pth
```

Both scripts auto-detect CUDA and fall back to CPU.

## Requirements

```
torch
torchvision
numpy
matplotlib
```
