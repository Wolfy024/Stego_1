"""Render auditable cover/stego and secret/recovery examples from a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from compare_checkpoints import load_manifest, load_rgb_image

from stego.metrics import psnr, ssim
from stego.training import load_model_from_checkpoint, select_device


def _rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor to a display-ready RGB array."""

    return (
        tensor.detach().float().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1
    ) / 2


def _configure_determinism(device: torch.device) -> None:
    torch.manual_seed(2026)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2026)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--manifest",
        default="results/eval_manifests/final.json",
        help="content-addressed cover/secret manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/qualitative_examples.png"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    if args.examples < 1:
        parser.error("--examples must be at least 1")

    device = select_device(args.device)
    _configure_determinism(device)
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    if config.model.payload_bits is not None:
        raise ValueError("the qualitative showcase requires an RGB-image checkpoint")

    pairs = load_manifest(args.manifest)[: args.examples]
    image_size = config.data.image_size
    cover = torch.stack(
        [load_rgb_image(pair.cover_path, image_size) for pair in pairs]
    ).to(device)
    secret = torch.stack(
        [load_rgb_image(pair.secret_path, image_size) for pair in pairs]
    ).to(device)

    with torch.inference_mode():
        outputs = model(cover, secret)
    stego = outputs["stego"].float()
    recovered = outputs["revealed_secret"].float()

    figure, axes = plt.subplots(
        len(pairs),
        4,
        figsize=(13.2, 3.35 * len(pairs)),
        squeeze=False,
    )
    column_titles = ("Cover", "Stego", "Secret", "Recovered secret")
    records: list[dict[str, float | str]] = []

    for row, pair in enumerate(pairs):
        cover_psnr = psnr(cover[row : row + 1], stego[row : row + 1]).item()
        cover_ssim = ssim(cover[row : row + 1], stego[row : row + 1]).item()
        secret_psnr = psnr(secret[row : row + 1], recovered[row : row + 1]).item()
        secret_ssim = ssim(secret[row : row + 1], recovered[row : row + 1]).item()
        records.append(
            {
                "pair_id": pair.identifier,
                "cover_psnr_db": cover_psnr,
                "cover_ssim": cover_ssim,
                "secret_psnr_db": secret_psnr,
                "secret_ssim": secret_ssim,
            }
        )

        images = (cover[row], stego[row], secret[row], recovered[row])
        subtitles = (
            pair.identifier,
            f"PSNR {cover_psnr:.2f} dB · SSIM {cover_ssim:.4f}",
            "hidden payload",
            f"PSNR {secret_psnr:.2f} dB · SSIM {secret_ssim:.4f}",
        )
        for column, (image, subtitle) in enumerate(zip(images, subtitles, strict=True)):
            axis = axes[row, column]
            axis.imshow(_rgb(image))
            if row == 0:
                axis.set_title(column_titles[column], fontsize=13, fontweight="bold", pad=10)
            axis.set_xlabel(subtitle, fontsize=9, labelpad=7)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#d1d5db")
                spine.set_linewidth(1.0)

    figure.suptitle(
        "Actual 256×256 image hiding and recovery",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.006,
        "First three pairs in results/eval_manifests/final.json · no cherry-picking",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0.01, 0.025, 0.99, 0.965), h_pad=1.4, w_pad=0.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
