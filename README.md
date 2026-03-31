# AFAS-Net — Attention-based Frequency Adaptive Steganography

> A deep-learning image steganography system that hides a full-colour **secret image** inside a **cover image** with imperceptible visual distortion, then recovers it with high fidelity.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.9-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

- **Adaptive Frequency Decomposition** – learnable convolutional frequency bands let the model decide where to embed secret data (high-frequency vs. low-frequency regions).
- **Multi-Scale Spatial Attention** – three-scale attention mechanism focuses embedding on perceptually less-sensitive areas of the cover image.
- **Adversarial Training** – a GAN discriminator enforces that stego images are statistically indistinguishable from clean cover images.
- **Perceptual Loss** – VGG-based perceptual similarity on both the stego and recovered secret keeps visual quality high.
- **Residual Embedding** – the encoder adds only a small perturbation (×0.1) on top of the cover, minimising artefacts.
- **Quality Metrics** – PSNR and SSIM are computed for both the cover→stego and secret→revealed pairs after every forward pass.

---

## 🏗️ Architecture

```
Cover Image ──┐
              ├──► EncoderNetwork ──► Stego Image ──► DecoderNetwork ──► Revealed Secret
Secret Image ─┘          │                  │
                  FreqDecomposition   DiscriminatorNetwork
                  MultiScaleAttention
```

| Module | File | Role |
|---|---|---|
| `AFASNet` | `AFASNet.py` | Top-level model wiring Encoder ↔ Decoder ↔ Discriminator |
| `EncoderNetwork` | `Modules/Encoder.py` | Hides secret into cover using frequency + attention |
| `DecoderNetwork` | `Modules/Decoder.py` | Recovers secret from stego image |
| `DiscriminatorNetwork` | `Modules/Discriminator.py` | PatchGAN-style classifier for adversarial training |
| `FrequencyDecomposition` | `Modules/FrequencyDecomposition.py` | Learnable multi-band frequency analysis |
| `MultiScaleAttention` | `Modules/MultiScaleAttention.py` | Spatial attention at scales 1×, 2×, 4× |
| `SteganographyDataset` | `Data/Dataset.py` | Pairs random cover + secret images for training |
| `evaluate_metrics` | `Modules/eval.py` | PSNR & SSIM for cover/secret fidelity |

---

## 📁 Project Structure

```
Stego_1/
├── AFASNet.py               # Top-level AFAS-Net model
├── main.py                  # Quick-start / smoke-test script
├── train.py                 # Training loop with GAN + perceptual losses
├── requirements.txt         # Python dependencies
├── Modules/
│   ├── Encoder.py
│   ├── Decoder.py
│   ├── Discriminator.py
│   ├── FrequencyDecomposition.py
│   ├── MultiScaleAttention.py
│   ├── ResidualBlock.py
│   └── eval.py
└── Data/
    ├── Dataset.py
    └── download_data.py
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.9 (CPU also works for inference)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Wolfy024/Stego_1.git
cd Stego_1

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Download the Dataset

The training script expects images under `Data/Animals-10/`. You can download the [Animals-10 dataset from Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10) and extract it there, or use the provided helper:

```bash
python Data/download_data.py
```

---

## 🔧 Usage

### Smoke Test (no data required)

Runs a single forward pass with random tensors to verify the model loads correctly:

```bash
python main.py
```

Expected output:
```
Forward pass successful!
Stego image shape: torch.Size([2, 3, 256, 256])
Revealed secret shape: torch.Size([2, 3, 256, 256])
Metrics: {'cover_psnr': ..., 'secret_psnr': ..., 'cover_ssim': ..., 'secret_ssim': ...}
```

### Training

Edit the dataset path in `main.py` if needed, then run:

```bash
python main.py
```

Or call the training function directly from your own script:

```python
from AFASNet import AFASNet
from train import train_afas_net
from torch.utils.data import DataLoader
from Data.Dataset import SteganographyDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = SteganographyDataset(root_dir="Data/Animals-10", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = AFASNet()
train_afas_net(model, dataloader, num_epochs=100)
```

Model checkpoints are saved every 10 epochs as `afas_net_epoch_<N>.pth`.

### Inference

```python
import torch
from AFASNet import AFASNet

model = AFASNet()
model.load_state_dict(torch.load("afas_net_epoch_100.pth"))
model.eval()

cover = ...   # torch.Tensor [B, 3, 256, 256], values in [-1, 1]
secret = ...  # torch.Tensor [B, 3, 256, 256], values in [-1, 1]

with torch.no_grad():
    outputs = model(cover, secret)

stego           = outputs['stego']            # cover with secret embedded
revealed_secret = outputs['revealed_secret']  # recovered secret image
```

---

## 📊 Training Details

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam (β₁=0.5, β₂=0.999) |
| Learning rate | 2 × 10⁻⁴ |
| Image size | 256 × 256 |
| Default batch size | 2 |
| Default epochs | 100 |
| Max training images | 6 000 |

**Loss components (Generator)**

| Loss | Weight |
|---|---|
| MSE cover reconstruction | 10 |
| MSE secret reconstruction | 10 |
| Perceptual cover loss | 0.1 |
| Perceptual secret loss | 0.1 |
| Adversarial (BCE) | 0.01 |
| Frequency regularisation | 0.1 |

---

## 📦 Dependencies

| Package | Version |
|---|---|
| `torch` | 2.8.0+cu129 |
| `torchvision` | 0.23.0+cu129 |
| `pillow` | 11.3.0 |
| `kaggle` | 1.7.4.5 |

---

## 📄 License

This project is released under the [MIT License](LICENSE).
