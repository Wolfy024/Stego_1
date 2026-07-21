"""U-Net steganography generator, keyless revealer, and PatchGAN critic."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm

from stego.config import ModelConfig
from stego.invertible import WaveletGatedInvertibleHider


def _conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    *,
    stride: int = 1,
    padding: int | None = None,
    use_spectral_norm: bool = True,
) -> nn.Conv2d:
    if padding is None:
        padding = kernel_size // 2
    layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    )
    return spectral_norm(layer) if use_spectral_norm else layer


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel then spatial attention)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        average = F.adaptive_avg_pool2d(x, 1)
        maximum = F.adaptive_max_pool2d(x, 1)
        channel_attention = torch.sigmoid(self.channel_mlp(average) + self.channel_mlp(maximum))
        x = x * channel_attention
        spatial_input = torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        return x * torch.sigmoid(self.spatial(spatial_input))


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        groups = min(32, out_channels)
        while out_channels % groups:
            groups -= 1
        self.body = nn.Sequential(
            _conv(
                in_channels,
                out_channels,
                stride=stride,
                use_spectral_norm=use_spectral_norm,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            _conv(out_channels, out_channels, use_spectral_norm=use_spectral_norm),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            CBAM(out_channels),
        )
        if in_channels == out_channels and stride == 1:
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = _conv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                use_spectral_norm=use_spectral_norm,
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x) + self.skip(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        use_spectral_norm: bool,
    ) -> None:
        super().__init__()
        self.project = _conv(in_channels, out_channels, use_spectral_norm=use_spectral_norm)
        self.fuse = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((self.project(x), skip), dim=1))


class UNet(nn.Module):
    """Three-level CBAM U-Net with a 512-channel bottleneck at base=64."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        base_channels: int,
        *,
        use_spectral_norm: bool,
        output_activation: Callable[[Tensor], Tensor] | None = torch.tanh,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (base_channels * factor for factor in (1, 2, 4, 8))
        self.stem = ConvBlock(input_channels, c1, use_spectral_norm=use_spectral_norm)
        self.down1 = ConvBlock(c1, c2, stride=2, use_spectral_norm=use_spectral_norm)
        self.down2 = ConvBlock(c2, c3, stride=2, use_spectral_norm=use_spectral_norm)
        self.down3 = ConvBlock(c3, c4, stride=2, use_spectral_norm=use_spectral_norm)
        self.bottleneck = ConvBlock(c4, c4, use_spectral_norm=use_spectral_norm)
        self.up3 = UpBlock(c4, c3, c3, use_spectral_norm=use_spectral_norm)
        self.up2 = UpBlock(c3, c2, c2, use_spectral_norm=use_spectral_norm)
        self.up1 = UpBlock(c2, c1, c1, use_spectral_norm=use_spectral_norm)
        self.output = nn.Conv2d(c1, output_channels, kernel_size=1)
        self.output_activation = output_activation

    def forward(self, x: Tensor) -> Tensor:
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        x = self.bottleneck(self.down3(s3))
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.output(x)
        return self.output_activation(x) if self.output_activation is not None else x


class PayloadExpander(nn.Module):
    """Arrange payload bits on a spatial grid that gives every bit an address."""

    def __init__(self, payload_bits: int) -> None:
        super().__init__()
        self.payload_bits = payload_bits
        self.rows = max(1, math.floor(math.sqrt(payload_bits)))
        self.columns = math.ceil(payload_bits / self.rows)

    def forward(self, payload: Tensor, spatial_size: tuple[int, int]) -> Tensor:
        if payload.ndim != 2 or payload.shape[1] != self.payload_bits:
            raise ValueError(
                f"payload must have shape [batch, {self.payload_bits}], got {payload.shape}"
            )
        cells = self.rows * self.columns
        centered = payload * 2.0 - 1.0
        if cells > self.payload_bits:
            centered = F.pad(centered, (0, cells - self.payload_bits))
        grid = centered.view(payload.shape[0], 1, self.rows, self.columns)
        return F.interpolate(grid, size=spatial_size, mode="nearest")


class HidingNetwork(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.max_residual = config.max_residual
        self.target_residual_rms = 2.0 * 10.0 ** (-config.target_cover_psnr / 20.0)
        self.payload_expander = (
            PayloadExpander(config.payload_bits) if config.payload_bits is not None else None
        )
        if config.payload_bits is None:
            preparation_channels = max(config.base_channels // 2, 16)
            self.secret_preparer: nn.Module | None = nn.Sequential(
                ConvBlock(
                    3,
                    preparation_channels,
                    use_spectral_norm=config.spectral_norm,
                ),
                ConvBlock(
                    preparation_channels,
                    preparation_channels,
                    use_spectral_norm=config.spectral_norm,
                ),
            )
            conditioning_channels = preparation_channels
        else:
            self.secret_preparer = None
            conditioning_channels = 1
        self.unet = UNet(
            input_channels=3 + conditioning_channels,
            output_channels=3,
            base_channels=config.base_channels,
            use_spectral_norm=config.spectral_norm,
        )

    def forward(self, cover: Tensor, secret: Tensor) -> tuple[Tensor, Tensor]:
        if self.payload_expander is not None:
            conditioning = self.payload_expander(secret, cover.shape[-2:])
        else:
            if cover.shape[:2] != secret.shape[:2]:
                raise ValueError(
                    "cover and secret must have matching batch/channel dimensions, "
                    f"got {cover.shape} and {secret.shape}"
                )
            conditioning = F.interpolate(
                secret,
                size=cover.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if self.secret_preparer is None:
                raise RuntimeError("image-payload mode requires the secret-preparation branch")
            conditioning = self.secret_preparer(conditioning)
        raw_residual = self.unet(torch.cat((cover, conditioning), dim=1)) * self.max_residual
        residual_rms = raw_residual.square().flatten(start_dim=1).mean(dim=1).sqrt()
        budget_scale = (self.target_residual_rms / residual_rms.clamp_min(1e-8)).clamp(max=1.0)
        residual = raw_residual * budget_scale[:, None, None, None]
        stego = torch.clamp(cover + residual, -1.0, 1.0)
        return stego, stego - cover


class PayloadDecoder(nn.Module):
    """Keyless 512-channel decoder that reads the addressed payload grid."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.payload_bits is None:
            raise ValueError("PayloadDecoder requires payload_bits")
        self.payload_bits = config.payload_bits
        self.rows = max(1, math.floor(math.sqrt(config.payload_bits)))
        self.columns = math.ceil(config.payload_bits / self.rows)
        c1, c2, c3, c4 = (config.base_channels * factor for factor in (1, 2, 4, 8))
        self.features = nn.Sequential(
            ConvBlock(3, c1, use_spectral_norm=config.spectral_norm),
            ConvBlock(c1, c2, stride=2, use_spectral_norm=config.spectral_norm),
            ConvBlock(c2, c3, stride=2, use_spectral_norm=config.spectral_norm),
            ConvBlock(c3, c4, stride=2, use_spectral_norm=config.spectral_norm),
            ConvBlock(c4, c4, use_spectral_norm=config.spectral_norm),
        )
        self.classifier = nn.Conv2d(c4, 1, kernel_size=1)

    def forward(self, stego: Tensor) -> Tensor:
        logit_map = self.classifier(self.features(stego))
        grid = F.adaptive_avg_pool2d(logit_map, (self.rows, self.columns))
        return grid.flatten(start_dim=1)[:, : self.payload_bits]


class RevealNetwork(nn.Module):
    """Keyless decoder: the stego image is its only input."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.secret_size = config.secret_size
        self.network: nn.Module
        if config.payload_bits is not None:
            self.network = PayloadDecoder(config)
        else:
            self.network = UNet(
                input_channels=3,
                output_channels=3,
                base_channels=config.base_channels,
                use_spectral_norm=config.spectral_norm,
            )

    def forward(self, stego: Tensor) -> Tensor:
        output = self.network(stego)
        if self.secret_size is not None and output.ndim == 4:
            output = F.interpolate(
                output,
                size=(self.secret_size, self.secret_size),
                mode="bilinear",
                align_corners=False,
            )
        return output


class PatchDiscriminator(nn.Module):
    """Spectrally-normalized PatchGAN critic returning logits."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            layers.extend(
                (
                    _conv(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        use_spectral_norm=True,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_channels = out_channels
        layers.append(_conv(in_channels, 1, kernel_size=3, padding=1, use_spectral_norm=True))
        self.network = nn.Sequential(*layers)

    def forward(self, image: Tensor) -> Tensor:
        return self.network(image)


class StegoGAN(nn.Module):
    """Complete model with separate generator and discriminator paths."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        if self.config.architecture == "invertible":
            if self.config.payload_bits is not None:
                raise ValueError("invertible architecture supports full-image payloads only")
            self.encoder: nn.Module = WaveletGatedInvertibleHider(
                blocks=self.config.invertible_blocks,
                growth_channels=self.config.coupling_channels,
                clamp=self.config.coupling_clamp,
                latent_noise_std=self.config.latent_noise_std,
                latent_seed=self.config.latent_seed,
                orthogonal_router=self.config.orthogonal_router,
            )
            self.decoder: nn.Module = nn.Identity()
        else:
            self.encoder = HidingNetwork(self.config)
            self.decoder = RevealNetwork(self.config)
        self.discriminator = PatchDiscriminator(self.config.base_channels)

    def forward(self, cover: Tensor, secret: Tensor) -> dict[str, Tensor]:
        if isinstance(self.encoder, WaveletGatedInvertibleHider):
            return self.encoder(cover, secret)
        stego, residual = self.encoder(cover, secret)
        revealed = self.decoder(stego)
        return {"stego": stego, "revealed_secret": revealed, "residual": residual}

    def reveal(self, stego: Tensor) -> Tensor:
        if isinstance(self.encoder, WaveletGatedInvertibleHider):
            return self.encoder.reveal(stego)
        return self.decoder(stego)

    def generator_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for parameter in (*self.encoder.parameters(), *self.decoder.parameters())
            if parameter.requires_grad
        ]

    @property
    def bottleneck_channels(self) -> int:
        return self.config.base_channels * 8
