"""Range-correct, batch-aware image quality metrics."""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F


def _per_image_mean(x: Tensor) -> Tensor:
    return x.flatten(start_dim=1).mean(dim=1)


def psnr(reference: Tensor, estimate: Tensor, data_range: float = 2.0) -> Tensor:
    """Peak signal-to-noise ratio for each image in a batch."""

    mse = _per_image_mean((reference - estimate).square())
    perfect = torch.full_like(mse, float("inf"))
    return torch.where(mse == 0, perfect, 10.0 * torch.log10(data_range**2 / mse))


def ssim(reference: Tensor, estimate: Tensor, data_range: float = 2.0) -> Tensor:
    """Channel-wise SSIM using the standard 11x11 Gaussian local window."""

    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have identical shapes")
    if reference.ndim != 4:
        raise ValueError("SSIM expects tensors shaped [batch, channels, height, width]")
    window_size = min(11, reference.shape[-2], reference.shape[-1])
    if window_size % 2 == 0:
        window_size -= 1
    coordinates = torch.arange(window_size, device=reference.device, dtype=reference.dtype)
    coordinates -= (window_size - 1) / 2
    gaussian = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    window_2d = torch.outer(gaussian, gaussian)
    channels = reference.shape[1]
    window = window_2d.expand(channels, 1, window_size, window_size)
    padding = window_size // 2

    def filter_image(image: Tensor) -> Tensor:
        padded = F.pad(image, (padding, padding, padding, padding), mode="reflect")
        return F.conv2d(padded, window, groups=channels)

    mu_x = filter_image(reference)
    mu_y = filter_image(estimate)
    sigma_x = filter_image(reference.square()) - mu_x.square()
    sigma_y = filter_image(estimate.square()) - mu_y.square()
    sigma_xy = filter_image(reference * estimate) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return _per_image_mean(score)


def extraction_similarity(reference: Tensor, estimate: Tensor, data_range: float = 2.0) -> Tensor:
    """MAE-derived fidelity percentage; not an exact-pixel accuracy metric."""

    similarity = 1.0 - _per_image_mean((reference - estimate).abs()) / data_range
    return similarity.clamp(0.0, 1.0) * 100.0


def image_metrics(reference: Tensor, estimate: Tensor, prefix: str) -> dict[str, Tensor]:
    return {
        f"{prefix}_psnr_db": psnr(reference, estimate),
        f"{prefix}_ssim": ssim(reference, estimate),
        f"{prefix}_similarity_pct": extraction_similarity(reference, estimate),
    }


def bit_accuracy(payload: Tensor, logits: Tensor) -> Tensor:
    """Exact bit recovery accuracy for each item in a batch, as a percentage."""

    if payload.shape != logits.shape:
        raise ValueError(f"payload/logit shape mismatch: {payload.shape} vs {logits.shape}")
    recovered = logits >= 0
    expected = payload >= 0.5
    return (recovered == expected).float().mean(dim=1) * 100.0


def real_jpeg_roundtrip(images: Tensor, quality: int, subsampling: int = 2) -> Tensor:
    """Round-trip a batch through Pillow/libjpeg for honest robustness evaluation."""

    if not 1 <= quality <= 100:
        raise ValueError("quality must be within [1, 100]")
    device, dtype = images.device, images.dtype
    output: list[Tensor] = []
    uint8_images = ((images.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).round().byte()
    for item in uint8_images:
        array = item.permute(1, 2, 0).numpy()
        buffer = io.BytesIO()
        Image.fromarray(array, mode="RGB").save(
            buffer, format="JPEG", quality=quality, subsampling=subsampling, optimize=False
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            restored = np.asarray(decoded.convert("RGB"), dtype=np.float32).copy()
        output.append(torch.from_numpy(restored).permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(output).to(device=device, dtype=dtype)
