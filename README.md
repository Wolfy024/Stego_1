# Wavelet-Gated Invertible Image Steganography

> Hide one full **256×256 RGB secret image** inside a **256×256 RGB cover**, then recover it from the stego image alone.

[![CI](https://github.com/Wolfy024/Stego_1/actions/workflows/ci.yml/badge.svg)](https://github.com/Wolfy024/Stego_1/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![image size](https://img.shields.io/badge/payload-256%C3%97256%20RGB-2563EB)](#payload)

The selected checkpoint clears both clean-channel quality goals on 200 held-out
cover/secret pairs:

| Path | PSNR ↑ | SSIM ↑ | MAE similarity ↑ |
|---|---:|---:|---:|
| Cover → stego | **38.55 dB** | **0.9631** | **99.15%** |
| Secret → recovered (clean) | **36.35 dB** | **0.9602** | **98.98%** |
| Secret → recovered (real JPEG QF-50) | 11.23 dB | 0.0093 | 75.87% |

“MAE similarity” is `100 × (1 − MAE / data_range)`; it is a fidelity score,
not exact-pixel extraction accuracy. The real-JPEG row is intentionally
prominent: this checkpoint is a strong **clean-channel** image hider, not a
compression-robust system.

![Architecture comparison](assets/architecture_comparison.png)

## What changed

The original CBAM U-Net baseline preserved the cover but plateaued at 16.15 dB
clean secret recovery. The selected model replaces that lossy encoder/decoder
path with a reversible mapping inspired by
[HiNet (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html):

```text
cover ── Haar DWT ──┐
                    ├─ 16 reversible coupling blocks ─ IWT ─ stego
secret ─ Haar DWT ──┘                              └──── latent

stego ─ Haar DWT ──┐
                    ├─ inverse coupling blocks ─ IWT ─ recovered secret
fixed latent z ─────┘
```

This repository adds a small, project-specific spin to that backbone:

- **CBAM-style coupling attention** reweights dense features inside each
  reversible subnet.
- **Sample-adaptive wavelet gates** independently modulate LL, HL, LH, and HH
  residual bands.
- Both modules are identity-initialized, so a compatible checkpoint can be
  imported without changing its initial output.
- A gates-only adaptation mode freezes the imported coupling backbone and
  updates 123,024 attention/gate parameters.
- Evaluation uses a fixed seeded Gaussian latent, so reveal is deterministic
  and requires no user-supplied key.

This is an engineering extension, not a claim of a new academic contribution.
The inference path has 4.17M parameters; the complete checkpoint is 4.87M
parameters including the optional PatchGAN critic.

## Payload

The primary task is full-image hiding, **not a 96-bit watermark**:

- cover: `[B, 3, 256, 256]`
- secret: `[B, 3, 256, 256]`
- stego: `[B, 3, 256, 256]`
- recovered secret: `[B, 3, 256, 256]`

That is a nominal raw payload of 1,572,864 bits per image
(`256 × 256 × 3 × 8`). Recovery is learned and lossy; the count is not a
claim of lossless capacity. The older bit-payload mode remains available as an
optional U-Net configuration.

## Results

### Benchmark protocol

- **Data discovered:** 5,000 COCO val2017 images + 100 DIV2K validation images
- **Deterministic sample:** 2,000 images, seed 2026
- **Split:** 1,800 train / 200 validation with no file overlap
- **Input and secret size:** 256×256 RGB
- **Aggregation:** per-image RGB PSNR/SSIM, then arithmetic mean over 200 pairs
- **JPEG evaluation:** actual Pillow/libjpeg encode/decode with 4:2:0
  subsampling—not the differentiable training approximation
- **Selected checkpoint:** official HiNet warm start + 100 local gates-only
  steps

The data audit found 5,100/5,100 readable images, no exact duplicate files, and
a disjoint split. Only 3 of the 200 sampled validation images are from DIV2K,
so the aggregate should not be presented as a DIV2K-specific result.

### Model comparison

| Model | Cover PSNR | Clean secret PSNR | QF-50 secret PSNR |
|---|---:|---:|---:|
| CBAM U-Net, 3,900 steps | **38.60** | 16.15 | **15.11** |
| Wavelet-gated INN, 100 gate steps | 38.55 | **36.35** | 11.23 |

Invertibility improves clean secret recovery by **20.19 dB** on the same split
while retaining essentially the same cover PSNR. It does not improve JPEG
robustness.

### Qualitative examples

The figure uses the first three validation pairs in deterministic order; they
were not cherry-picked. The residual view is auto-scaled for visibility.

![Qualitative clean and JPEG examples](assets/qualitative_examples.png)

### JPEG stress test

![JPEG robustness sweep](assets/jpeg_robustness.png)

All tested real-JPEG qualities collapse recovery to roughly 11.2 dB. A robust
version would need to move away from brittle full-band invertibility—for
example, a redundancy/error-correction path or a robust invertible objective
such as the direction explored by
[RIIS (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Robust_Invertible_Image_Steganography_CVPR_2022_paper.html).

Machine-readable evidence lives in
[`results/benchmark_256.json`](results/benchmark_256.json), and the independent
checks are recorded in
[`results/validation_report.md`](results/validation_report.md).

## Reproduce

### 1. Install

```bash
git clone https://github.com/Wolfy024/Stego_1.git
cd Stego_1
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the correct PyTorch build for your CUDA version first if the default
wheel is not appropriate. CPU smoke tests work; training the 16-block model is
GPU-oriented.

### 2. Download data

```bash
python scripts/download_datasets.py coco-val2017 div2k-valid
python scripts/profile_data.py \
  --data-dir data/coco/val2017 data/div2k/DIV2K_valid_HR
```

The helper uses the official
[COCO](https://cocodataset.org/#download) and
[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) archives and stores them
under ignored `data/` paths.

### 3. Import the disclosed warm start

Download `model.pt` from the
[official HiNet trained-model folder](https://drive.google.com/drive/folders/1l3XBFYPMaNFdvCWyOHfB2qIPkpjIxZgE?usp=sharing)
to `external/hinet-model/model.pt`, then run:

```bash
python scripts/import_hinet_checkpoint.py \
  --source external/hinet-model/model.pt \
  --output-dir runs/invertible_warmstart
```

The importer maps 480 compatible dense-convolution tensors. The benchmark used
official repository commit
`5c2682a6be3fc52354b816c0269c05061e30b998` and source checkpoint SHA-256
`d89a83e0e549e9bddde301d631b8614cc724c8d9f11feaf2221791aa02c122a7`.
The upstream checkpoint is not redistributed here; obtain it from its authors
and follow their terms.

### 4. Adapt only the new modules

```bash
python -m stego.cli train \
  --resume runs/invertible_warmstart/latest.pt \
  --data-dir data/coco/val2017 data/div2k/DIV2K_valid_HR \
  --output-dir runs/invertible_gates_only \
  --max-steps 100 \
  --cover-weight 20 \
  --secret-weight 5 \
  --low-frequency-weight 10 \
  --gates-only
```

### 5. Evaluate, render, and validate

```bash
python -m stego.cli evaluate \
  --checkpoint runs/invertible_gates_only/latest.pt \
  --data-dir data/coco/val2017 data/div2k/DIV2K_valid_HR \
  --jpeg-qualities 50 60 70 80 90 \
  --output results/reproduced_benchmark_256.json

python scripts/render_examples.py \
  --checkpoint runs/invertible_gates_only/latest.pt \
  --data-dir data/coco/val2017 data/div2k/DIV2K_valid_HR

python scripts/plot_results.py

python scripts/validate_results.py \
  --checkpoint runs/invertible_gates_only/latest.pt \
  --warmstart runs/invertible_warmstart/latest.pt \
  --history runs/invertible_gates_only/history.csv
```

## Quick checks

```bash
python -m stego.cli smoke --device cpu
ruff check .
pytest
```

The 18 fast CPU tests cover Haar round trips, invertible forward/reverse paths,
attention identity initialization, data pairing, range-correct PSNR/SSIM,
real and differentiable JPEG paths, and configuration validation.
`validate_results.py` separately checks the selected checkpoint's 256×256
tensor shapes, headline metrics, gate updates, and training stability.

## Repository map

```text
configs/                     reproducible U-Net, watermark, and INN configs
stego/
  invertible.py              Haar transform, gated coupling blocks, inverse
  models.py                  selectable INN and CBAM U-Net/PatchGAN models
  training.py                AMP training, checkpointing, real-JPEG evaluation
  metrics.py                 batch-aware PSNR, SSIM, and similarity metrics
  data.py                    deterministic cover/secret pairing
  cli.py                     smoke, train, and evaluate entry points
scripts/
  download_datasets.py       credential-free official dataset downloader
  import_hinet_checkpoint.py explicit upstream tensor mapping
  plot_results.py            source-backed README charts
  render_examples.py         deterministic qualitative panel
  validate_results.py        independent evidence reconciliation
results/                     committed benchmark, audit, and chart provenance
tests/                       fast CPU regression tests
```

## Honest résumé version

- Built a 4.17M-parameter wavelet-gated invertible image-hiding model for
  full 256×256 RGB payloads, adding CBAM coupling attention and
  sample-adaptive Haar-band gates to a disclosed HiNet warm start.
- Reached **38.55 dB cover PSNR** and **36.35 dB recovered-secret PSNR**
  (SSIM 0.963/0.960) over 200 held-out COCO/DIV2K pairs.
- Improved clean secret recovery by **20.19 dB** over the same-split CBAM U-Net
  baseline; built real-JPEG evaluation that exposed the current model's
  compression limitation.

Do not describe 98.98% similarity as “99% extraction accuracy,” and do not
claim JPEG robustness from this checkpoint.

## Limitations and security

- Clean recovery assumes an unchanged tensor or lossless image path.
- JPEG, resizing, cropping, and other distribution shifts are not currently
  supported.
- “Keyless” means no user-supplied decoding key; it does **not** provide
  confidentiality, authentication, or cryptographic security.
- Results use one seeded split and no multi-seed confidence intervals.
- The final backbone is warm-started from public HiNet weights, not trained
  from scratch.
- The comparison is an internal architecture ablation, not an independent
  reproduction of every published baseline.

## Acknowledgment

The reversible coupling equations, Haar-domain construction, and compatible
backbone weights come from:

> Jing et al., “HiNet: Deep Image Hiding by Invertible Network,” ICCV 2021.

See the
[paper](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html)
and
[official implementation](https://github.com/TomTomTommi/HiNet).
