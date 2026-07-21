"""Differentiable JPEG approximation used for robustness training."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_LUMA_QUANTIZATION = (
    (16, 11, 10, 16, 24, 40, 51, 61),
    (12, 12, 14, 19, 26, 58, 60, 55),
    (14, 13, 16, 24, 40, 57, 69, 56),
    (14, 17, 22, 29, 51, 87, 80, 62),
    (18, 22, 37, 56, 68, 109, 103, 77),
    (24, 35, 55, 64, 81, 104, 113, 92),
    (49, 64, 78, 87, 103, 121, 120, 101),
    (72, 92, 95, 98, 112, 100, 103, 99),
)

_CHROMA_QUANTIZATION = (
    (17, 18, 24, 47, 99, 99, 99, 99),
    (18, 21, 26, 66, 99, 99, 99, 99),
    (24, 26, 56, 99, 99, 99, 99, 99),
    (47, 66, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
)


def _dct_matrix() -> Tensor:
    matrix = torch.empty(8, 8)
    for frequency in range(8):
        scale = math.sqrt(1.0 / 8.0) if frequency == 0 else math.sqrt(2.0 / 8.0)
        for position in range(8):
            matrix[frequency, position] = scale * math.cos(
                math.pi * (2 * position + 1) * frequency / 16
            )
    return matrix


def _ste_round(x: Tensor) -> Tensor:
    """Round in the forward pass while using an identity backward pass."""

    return x + (torch.round(x) - x).detach()


class DifferentiableJPEG(nn.Module):
    """Differentiable JPEG approximation operating on tensors in ``[-1, 1]``.

    The layer models color conversion, 8x8 DCT, quality-scaled quantization,
    optional chroma prefiltering, and inverse DCT. Straight-through rounding
    gives useful gradients. Final evaluation should still use a real JPEG
    codec because this approximation deliberately omits entropy coding and
    does not exactly reproduce a codec's chroma resampling.
    """

    def __init__(
        self,
        quality: int = 75,
        differentiable: bool = True,
        chroma_subsampling: bool = True,
    ) -> None:
        super().__init__()
        if not 1 <= quality <= 100:
            raise ValueError("quality must be within [1, 100]")
        self.quality = quality
        self.differentiable = differentiable
        self.chroma_subsampling = chroma_subsampling
        self.register_buffer("dct", _dct_matrix(), persistent=False)
        tables = torch.tensor((_LUMA_QUANTIZATION, _CHROMA_QUANTIZATION, _CHROMA_QUANTIZATION))
        self.register_buffer("base_tables", tables.float(), persistent=False)

    def _quantization_tables(self, quality: int, dtype: torch.dtype) -> Tensor:
        scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
        tables = torch.floor((self.base_tables * scale + 50.0) / 100.0)
        return tables.clamp_(1.0, 255.0).to(dtype=dtype).view(1, 3, 1, 1, 8, 8)

    @staticmethod
    def _rgb_to_ycbcr(image: Tensor) -> Tensor:
        red, green, blue = image.unbind(dim=1)
        y = 0.299 * red + 0.587 * green + 0.114 * blue
        cb = -0.168736 * red - 0.331264 * green + 0.5 * blue + 0.5
        cr = 0.5 * red - 0.418688 * green - 0.081312 * blue + 0.5
        return torch.stack((y, cb, cr), dim=1)

    @staticmethod
    def _ycbcr_to_rgb(image: Tensor) -> Tensor:
        y, cb, cr = image.unbind(dim=1)
        cb = cb - 0.5
        cr = cr - 0.5
        red = y + 1.402 * cr
        green = y - 0.344136 * cb - 0.714136 * cr
        blue = y + 1.772 * cb
        return torch.stack((red, green, blue), dim=1)

    @staticmethod
    def _blockify(image: Tensor) -> Tensor:
        batch, channels, height, width = image.shape
        return (
            image.view(batch, channels, height // 8, 8, width // 8, 8)
            .permute(0, 1, 2, 4, 3, 5)
            .contiguous()
        )

    @staticmethod
    def _deblockify(blocks: Tensor) -> Tensor:
        batch, channels, block_rows, block_columns, _, _ = blocks.shape
        return (
            blocks.permute(0, 1, 2, 4, 3, 5)
            .contiguous()
            .view(batch, channels, block_rows * 8, block_columns * 8)
        )

    def forward(self, image: Tensor, quality: int | None = None) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [batch, 3, height, width]")
        quality = self.quality if quality is None else quality
        if not 1 <= quality <= 100:
            raise ValueError("quality must be within [1, 100]")

        original_height, original_width = image.shape[-2:]
        pad_height = (-original_height) % 8
        pad_width = (-original_width) % 8
        image = F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")

        ycbcr = self._rgb_to_ycbcr((image + 1.0) * 0.5)
        if self.chroma_subsampling:
            chroma = F.avg_pool2d(ycbcr[:, 1:], kernel_size=2, stride=2)
            chroma = F.interpolate(
                chroma,
                size=ycbcr.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            ycbcr = torch.cat((ycbcr[:, :1], chroma), dim=1)
        ycbcr = ycbcr * 255.0 - 128.0
        blocks = self._blockify(ycbcr)
        dct_blocks = torch.matmul(torch.matmul(self.dct, blocks), self.dct.transpose(0, 1))
        tables = self._quantization_tables(quality, image.dtype)
        scaled = dct_blocks / tables
        quantized = _ste_round(scaled) if self.differentiable else torch.round(scaled)
        reconstructed = torch.matmul(
            torch.matmul(self.dct.transpose(0, 1), quantized * tables), self.dct
        )
        ycbcr = (self._deblockify(reconstructed) + 128.0) / 255.0
        rgb = self._ycbcr_to_rgb(ycbcr).clamp(0.0, 1.0)
        return (rgb[:, :, :original_height, :original_width] * 2.0 - 1.0).contiguous()
