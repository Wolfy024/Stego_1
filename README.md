# Wavelet-Gated Image Steganography

Hide a complete **256×256 RGB image** inside another 256×256 RGB image and
recover it from the stego image alone.

[![CI](https://github.com/Wolfy024/Stego_1/actions/workflows/ci.yml/badge.svg)](https://github.com/Wolfy024/Stego_1/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)

## Actual hiding and recovery

![Three actual cover, stego, secret, and recovered-secret image pairs](assets/qualitative_examples.png)

These are direct outputs from the current checkpoint on the **first three
pairs** in the content-addressed final manifest. The examples are deterministic
and were not selected by visual quality. Each stego and recovery panel includes
its own PSNR and SSIM.

## Results

Evaluated in FP32 on 200 fixed cover/secret pairs: 400 unique images from COCO
val2017 and DIV2K validation, with no image reused across roles.

| Measurement | Result |
|---|---:|
| Cover → stego PSNR | **38.762 dB** |
| Cover → stego SSIM | **0.9659** |
| Secret → recovered PSNR | **37.116 dB** |
| Secret → recovered SSIM | **0.9662** |
| Payload | **256×256 RGB image** |
| Evaluation size | **200 pairs / 400 unique images** |

Pair order, image hashes, checkpoint hashes, aggregate metrics, and per-pair
records are committed in
[`results/paired_final.json`](results/paired_final.json). The fixed cohort is
defined in
[`results/eval_manifests/final.json`](results/eval_manifests/final.json).

## Architecture

~~~text
  256×256 cover ── Haar DWT ──┐
                              ├─ 16 gated affine coupling blocks ─ IWT ─ stego
 256×256 secret ─ Haar DWT ──┘                                  └──── latent

       stego ──── Haar DWT ──┐
                             ├─ inverse coupling blocks ─ IWT ─ recovered secret
 seeded latent ──────────────┘
~~~

The inference network has **4.17M parameters** and combines:

- invertible affine coupling blocks for joint hiding and recovery;
- Haar-wavelet processing across LL, HL, LH, and HH frequency bands;
- CBAM-style attention inside the coupling subnets;
- sample-adaptive wavelet gates that modulate each frequency band;
- a fixed seeded latent for deterministic, keyless reveal.

The attention and wavelet-gate adapter contains **123,024 trainable
parameters**, allowing lightweight adaptation while the coupling backbone is
frozen.

## Model API

Inputs are normalized to `[-1, 1]` and use shape `[B, 3, 256, 256]`.

~~~python
import torch

from stego.training import load_model_from_checkpoint

device = torch.device("cuda")
model, config = load_model_from_checkpoint(
    "runs/invertible_gates_only/latest.pt",
    device,
)

# cover and secret: [B, 3, 256, 256], normalized to [-1, 1]
with torch.inference_mode():
    outputs = model(cover, secret)
    stego = outputs["stego"]
    recovered_secret = model.reveal(stego)
~~~

The decoder receives only the stego tensor. No secret image or user-supplied
decoding key is passed to `reveal`.

## Setup

~~~bash
git clone https://github.com/Wolfy024/Stego_1.git
cd Stego_1

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

Install the PyTorch build matching your CUDA environment before the editable
install when needed.

### Download the evaluation data

~~~bash
python scripts/download_datasets.py coco-val2017 div2k-valid
~~~

The downloader uses the official COCO and DIV2K archives and stores them under
ignored `data/` paths.

### Prepare the checkpoint

Place the HiNet authors' `model.pt` in
`external/hinet-model/model.pt`, then map the compatible tensors:

~~~bash
python scripts/import_hinet_checkpoint.py \
  --source external/hinet-model/model.pt \
  --output-dir runs/invertible_warmstart
~~~

Adapt the attention and wavelet gates:

~~~bash
python -m stego.cli train \
  --warm-start runs/invertible_warmstart/latest.pt \
  --data-dir data/coco/val2017 data/div2k/DIV2K_valid_HR \
  --output-dir runs/invertible_gates_only \
  --max-steps 100 \
  --cover-weight 20 \
  --secret-weight 5 \
  --low-frequency-weight 10 \
  --gates-only
~~~

### Recreate the image showcase

~~~bash
python scripts/render_examples.py \
  --checkpoint runs/invertible_gates_only/latest.pt \
  --manifest results/eval_manifests/final.json \
  --output assets/qualitative_examples.png \
  --device cuda \
  --examples 3
~~~

The renderer validates the manifest and source-image hashes before inference,
uses deterministic FP32 settings, and prints each displayed pair's metrics.

## Validation

~~~bash
ruff check .
pytest
python -m stego.cli smoke --architecture invertible --device cpu
~~~

The repository includes 55 tests covering Haar round trips, forward/reverse
consistency, deterministic latent behavior, attention and gate initialization,
metric correctness, dataset pairing, checkpoint compatibility, and evidence
generation.

## Repository layout

~~~text
stego/
  invertible.py       Haar transform and gated invertible coupling network
  models.py           image-hiding model and discriminator
  training.py         training, evaluation, and checkpoint utilities
  metrics.py          PSNR, SSIM, and image-quality metrics
  data.py             deterministic image loading and pairing
  cli.py              smoke, train, and evaluate commands
scripts/
  render_examples.py  hash-verified qualitative output renderer
  compare_checkpoints.py
  create_eval_manifests.py
  import_hinet_checkpoint.py
  download_datasets.py
results/               fixed manifests and machine-readable evaluation evidence
tests/                 CPU-friendly regression and evidence tests
~~~

## Current limitations

- Recovery is designed for a clean or lossless stego-image path.
- JPEG recompression, resizing, and cropping are not supported.
- Keyless decoding is not encryption or authentication.
- The reported aggregate is one fixed 200-pair cohort and one adapted
  checkpoint.

## Acknowledgment

The invertible wavelet backbone and initialization weights follow
[HiNet: Deep Image Hiding by Invertible Network](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html).
Checkpoint weights are not redistributed; obtain them from the
[authors' repository](https://github.com/TomTomTommi/HiNet) and follow their
terms.

## License

Project code is available under the [MIT License](LICENSE).
