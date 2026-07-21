"""Render deterministic qualitative examples from a trained checkpoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from stego.data import ImagePairDataset, discover_images, split_images
from stego.metrics import psnr, real_jpeg_roundtrip, ssim
from stego.training import load_model_from_checkpoint, select_device


def _validation_dataset(roots: list[str], config) -> ImagePairDataset:
    images = discover_images(roots)
    if config.data.max_images is not None and len(images) > config.data.max_images:
        generator = torch.Generator().manual_seed(config.data.seed)
        order = torch.randperm(len(images), generator=generator).tolist()
        images = [images[index] for index in order[: config.data.max_images]]
    _, validation = split_images(images, config.data.train_fraction, config.data.seed)
    return ImagePairDataset(
        validation,
        config.data.image_size,
        augment=False,
        seed=config.data.seed + 1,
        payload_bits=None,
        secret_image_size=config.model.secret_size,
    )


def _rgb(tensor: torch.Tensor) -> np.ndarray:
    return ((tensor.detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1) / 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("assets/qualitative_examples.png"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    random.seed(2026)
    torch.manual_seed(2026)
    device = select_device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    dataset = _validation_dataset(args.data_dir, config)
    indices = list(range(min(args.examples, len(dataset))))
    pairs = [dataset[index] for index in indices]
    cover = torch.stack([pair[0] for pair in pairs]).to(device)
    secret = torch.stack([pair[1] for pair in pairs]).to(device)
    with torch.inference_mode():
        outputs = model(cover, secret)
        jpeg = real_jpeg_roundtrip(outputs["stego"].float(), quality=50)
        recovered_q50 = model.reveal(jpeg)

    columns = ["Cover", "Secret", "Stego", "Residual energy", "Recovered", "QF-50 recovered"]
    figure, axes = plt.subplots(len(indices), len(columns), figsize=(15, 2.75 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row in range(len(indices)):
        clean_psnr = psnr(secret[row : row + 1], outputs["revealed_secret"][row : row + 1]).item()
        clean_ssim = ssim(secret[row : row + 1], outputs["revealed_secret"][row : row + 1]).item()
        q50_psnr = psnr(secret[row : row + 1], recovered_q50[row : row + 1]).item()
        cover_psnr = psnr(cover[row : row + 1], outputs["stego"][row : row + 1]).item()
        images = [
            _rgb(cover[row]),
            _rgb(secret[row]),
            _rgb(outputs["stego"][row]),
            outputs["residual"][row].detach().float().abs().mean(0).cpu().numpy(),
            _rgb(outputs["revealed_secret"][row]),
            _rgb(recovered_q50[row]),
        ]
        subtitles = [
            f"pair #{indices[row]}",
            "ground truth",
            f"cover PSNR {cover_psnr:.1f} dB",
            "mean |Δ| (auto-scaled)",
            f"{clean_psnr:.1f} dB · SSIM {clean_ssim:.2f}",
            f"{q50_psnr:.1f} dB",
        ]
        for column, (image, subtitle) in enumerate(zip(images, subtitles, strict=True)):
            axis = axes[row, column]
            axis.imshow(image, cmap="magma" if column == 3 else None)
            axis.set_title(f"{columns[column]}\n{subtitle}", fontsize=9)
            axis.axis("off")
    figure.suptitle(
        "Deterministic held-out 256×256 examples (first three validation pairs)",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.01, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
