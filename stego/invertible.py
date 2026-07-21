"""Wavelet-gated invertible network for full-image steganography.

The reversible coupling equations and Haar-domain design are inspired by
HiNet (ICCV 2021). This implementation adds CBAM-style feature attention and
sample-adaptive wavelet-band gates inside each dense coupling subnet.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class HaarWavelet(nn.Module):
    """Exactly invertible 2D Haar transform implemented without parameters."""

    @staticmethod
    def encode(image: Tensor) -> Tensor:
        if image.shape[-2] % 2 or image.shape[-1] % 2:
            raise ValueError("Haar transform requires even spatial dimensions")
        top = image[:, :, 0::2, :] * 0.5
        bottom = image[:, :, 1::2, :] * 0.5
        x1 = top[:, :, :, 0::2]
        x2 = bottom[:, :, :, 0::2]
        x3 = top[:, :, :, 1::2]
        x4 = bottom[:, :, :, 1::2]
        low_low = x1 + x2 + x3 + x4
        high_low = -x1 - x2 + x3 + x4
        low_high = -x1 + x2 - x3 + x4
        high_high = x1 - x2 - x3 + x4
        return torch.cat((low_low, high_low, low_high, high_high), dim=1)

    @staticmethod
    def decode(coefficients: Tensor) -> Tensor:
        if coefficients.shape[1] % 4:
            raise ValueError("Haar coefficients must have a channel count divisible by four")
        channels = coefficients.shape[1] // 4
        x1, x2, x3, x4 = coefficients.split(channels, dim=1)
        x1, x2, x3, x4 = (value * 0.5 for value in (x1, x2, x3, x4))
        output = coefficients.new_empty(
            coefficients.shape[0],
            channels,
            coefficients.shape[2] * 2,
            coefficients.shape[3] * 2,
        )
        output[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        output[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        output[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        output[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return output

    def forward(self, image: Tensor, *, reverse: bool = False) -> Tensor:
        return self.decode(image) if reverse else self.encode(image)


class CouplingAttention(nn.Module):
    """Lightweight CBAM-style channel and spatial attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 16, 8)
        self.channel = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        nn.init.zeros_(self.channel[-1].weight)
        nn.init.zeros_(self.channel[-1].bias)
        nn.init.zeros_(self.spatial.weight)
        nn.init.zeros_(self.spatial.bias)

    def forward(self, features: Tensor) -> Tensor:
        average = F.adaptive_avg_pool2d(features, 1)
        maximum = F.adaptive_max_pool2d(features, 1)
        channel_gate = 2.0 * torch.sigmoid(self.channel(average) + self.channel(maximum))
        features = features * channel_gate
        spatial = torch.cat(
            (features.mean(dim=1, keepdim=True), features.amax(dim=1, keepdim=True)),
            dim=1,
        )
        return features * (2.0 * torch.sigmoid(self.spatial(spatial)))


class WaveletBandGate(nn.Module):
    """Sample-adaptive residual gate for the four Haar frequency bands."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels % 4:
            raise ValueError("wavelet-band gating requires a channel count divisible by four")
        self.channels_per_band = channels // 4
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, residual: Tensor) -> Tensor:
        batch, _, height, width = residual.shape
        bands = residual.view(batch, 4, self.channels_per_band, height, width)
        descriptor = bands.abs().mean(dim=(2, 3, 4))
        gates = 1.0 + 0.5 * torch.tanh(self.mlp(descriptor))
        return (bands * gates[:, :, None, None, None]).flatten(1, 2)


class AttentiveDenseSubnet(nn.Module):
    """Dense coupling function with CBAM and explicit wavelet-band gating."""

    def __init__(self, input_channels: int, output_channels: int, growth_channels: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for index in range(4):
            self.layers.append(
                nn.Conv2d(input_channels + index * growth_channels, growth_channels, 3, padding=1)
            )
        dense_channels = input_channels + 4 * growth_channels
        self.attention = CouplingAttention(dense_channels)
        self.output = nn.Conv2d(dense_channels, output_channels, 3, padding=1)
        self.band_gate = WaveletBandGate(output_channels)
        self.activation = nn.LeakyReLU(0.01, inplace=True)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        features = [inputs]
        for layer in self.layers:
            features.append(self.activation(layer(torch.cat(features, dim=1))))
        dense = self.attention(torch.cat(features, dim=1))
        return self.band_gate(self.output(dense))


class InvertibleCouplingBlock(nn.Module):
    """Affine-additive coupling block with an analytic inverse."""

    def __init__(self, channels: int = 12, growth_channels: int = 32, clamp: float = 2.0) -> None:
        super().__init__()
        self.channels = channels
        self.clamp = clamp
        self.phi = AttentiveDenseSubnet(channels, channels, growth_channels)
        self.rho = AttentiveDenseSubnet(channels, channels, growth_channels)
        self.eta = AttentiveDenseSubnet(channels, channels, growth_channels)

    def _scale(self, values: Tensor) -> Tensor:
        return torch.exp(self.clamp * 2.0 * (torch.sigmoid(values) - 0.5))

    def forward(self, inputs: Tensor, *, reverse: bool = False) -> Tensor:
        first, second = inputs.split(self.channels, dim=1)
        if not reverse:
            first_out = first + self.phi(second)
            second_out = self._scale(self.rho(first_out)) * second + self.eta(first_out)
        else:
            second_out = (second - self.eta(first)) / self._scale(self.rho(first))
            first_out = first - self.phi(second_out)
        return torch.cat((first_out, second_out), dim=1)


class InvertibleFlow(nn.Module):
    def __init__(self, blocks: int, growth_channels: int, clamp: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            InvertibleCouplingBlock(12, growth_channels, clamp) for _ in range(blocks)
        )

    def forward(self, inputs: Tensor, *, reverse: bool = False) -> Tensor:
        blocks = reversed(self.blocks) if reverse else self.blocks
        for block in blocks:
            inputs = block(inputs, reverse=reverse)
        return inputs


class WaveletGatedInvertibleHider(nn.Module):
    """Bidirectional cover/secret mapper; revealing requires only the stego image."""

    def __init__(
        self,
        *,
        blocks: int = 12,
        growth_channels: int = 32,
        clamp: float = 2.0,
        latent_noise_std: float = 1.0,
        latent_seed: int = 2026,
    ) -> None:
        super().__init__()
        self.wavelet = HaarWavelet()
        self.flow = InvertibleFlow(blocks, growth_channels, clamp)
        self.latent_noise_std = latent_noise_std
        self.latent_seed = latent_seed

    def conceal(self, cover: Tensor, secret: Tensor) -> dict[str, Tensor]:
        if cover.shape != secret.shape:
            raise ValueError(
                f"cover/secret shapes must match, got {cover.shape} and {secret.shape}"
            )
        cover_unit = (cover + 1.0) * 0.5
        secret_unit = (secret + 1.0) * 0.5
        cover_wavelet = self.wavelet(cover_unit)
        secret_wavelet = self.wavelet(secret_unit)
        transformed = self.flow(torch.cat((cover_wavelet, secret_wavelet), dim=1))
        stego_wavelet, latent = transformed.chunk(2, dim=1)
        stego_unit = self.wavelet(stego_wavelet, reverse=True).clamp(0.0, 1.0)
        stego = stego_unit * 2.0 - 1.0
        return {
            "stego": stego,
            "residual": stego - cover,
            "latent": latent,
            "cover_low": cover_wavelet[:, :3],
            "stego_low": stego_wavelet[:, :3],
        }

    def reveal(self, stego: Tensor) -> Tensor:
        stego_wavelet = self.wavelet((stego + 1.0) * 0.5)
        latent = torch.zeros_like(stego_wavelet)
        if self.training and self.latent_noise_std > 0:
            latent.normal_(std=self.latent_noise_std)
        elif self.latent_noise_std > 0:
            generator = torch.Generator(device=stego.device).manual_seed(self.latent_seed)
            latent.normal_(std=self.latent_noise_std, generator=generator)
        reconstructed = self.flow(torch.cat((stego_wavelet, latent), dim=1), reverse=True)
        secret_wavelet = reconstructed[:, 12:]
        secret_unit = self.wavelet(secret_wavelet, reverse=True).clamp(0.0, 1.0)
        return secret_unit * 2.0 - 1.0

    def forward(self, cover: Tensor, secret: Tensor) -> dict[str, Tensor]:
        outputs = self.conceal(cover, secret)
        outputs["revealed_secret"] = self.reveal(outputs["stego"])
        return outputs
