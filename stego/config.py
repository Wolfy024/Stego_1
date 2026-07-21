"""Typed experiment configuration with JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_REMOVED_NOOP_FIELDS: dict[str, dict[str, object]] = {
    "model": {
        "latent_calibration": False,
        "latent_predictor_channels": 0,
        "latent_predictor_blocks": 3,
    },
    "loss": {"latent_consistency": 0.0},
    "train": {"latent_only": False},
}


def _drop_removed_noops(section: str, values: object) -> dict[str, Any]:
    """Accept old checkpoints only when removed experimental options were disabled."""

    if not isinstance(values, dict):
        raise ValueError(f"configuration section {section!r} must be an object")
    cleaned = dict(values)
    for name, disabled_value in _REMOVED_NOOP_FIELDS.get(section, {}).items():
        if name not in cleaned:
            continue
        value = cleaned.pop(name)
        if value != disabled_value:
            raise ValueError(
                f"removed experimental option {section}.{name} is unsupported "
                f"unless it has its disabled value {disabled_value!r}"
            )
    return cleaned


@dataclass(slots=True)
class ModelConfig:
    """Model architecture parameters.

    ``base_channels=64`` produces the advertised 512-channel U-Net bottleneck.
    Smaller values are useful for smoke tests and constrained hardware.
    """

    architecture: str = "unet"
    base_channels: int = 64
    max_residual: float = 0.10
    target_cover_psnr: float = 38.5
    spectral_norm: bool = False
    payload_bits: int | None = None
    secret_size: int | None = 256
    invertible_blocks: int = 12
    coupling_channels: int = 32
    coupling_clamp: float = 2.0
    latent_noise_std: float = 1.0
    latent_seed: int = 2026
    orthogonal_router: bool = False

    def validate(self) -> None:
        if self.architecture not in {"unet", "invertible"}:
            raise ValueError("architecture must be 'unet' or 'invertible'")
        if self.base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        if not 0.0 < self.max_residual <= 0.25:
            raise ValueError("max_residual must be in (0, 0.25]")
        if not 20.0 <= self.target_cover_psnr <= 60.0:
            raise ValueError("target_cover_psnr must be within [20, 60] dB")
        if self.payload_bits is not None and self.payload_bits < 8:
            raise ValueError("payload_bits must be at least 8 or null for image-payload mode")
        if self.secret_size is not None and self.secret_size < 8:
            raise ValueError("secret_size must be at least 8 or null")
        if self.invertible_blocks < 1 or self.coupling_channels < 8:
            raise ValueError("invertible_blocks and coupling_channels are too small")
        if not 0.1 <= self.coupling_clamp <= 5.0:
            raise ValueError("coupling_clamp must be within [0.1, 5.0]")
        if self.latent_noise_std < 0:
            raise ValueError("latent_noise_std cannot be negative")
        if self.latent_seed < 0:
            raise ValueError("latent_seed cannot be negative")
        if self.architecture != "invertible" and self.orthogonal_router:
            raise ValueError("orthogonal_router requires invertible architecture")


@dataclass(slots=True)
class DataConfig:
    image_size: int = 256
    train_fraction: float = 0.9
    max_images: int | None = None
    workers: int = 4
    seed: int = 2026

    def validate(self) -> None:
        if self.image_size < 32 or self.image_size % 8:
            raise ValueError("image_size must be >= 32 and divisible by 8")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if self.workers < 0:
            raise ValueError("workers cannot be negative")


@dataclass(slots=True)
class LossConfig:
    cover: float = 1.0
    secret: float = 2.0
    robust_secret: float = 2.0
    adversarial: float = 0.01
    residual_tv: float = 0.02
    low_frequency: float = 0.0

    def validate(self) -> None:
        weights = asdict(self)
        negative = [name for name, value in weights.items() if value < 0.0]
        if negative:
            raise ValueError(f"loss weights cannot be negative: {', '.join(negative)}")


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 2e-4
    discriminator_learning_rate: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    grad_clip: float = 5.0
    jpeg_quality_min: int = 50
    jpeg_quality_max: int = 95
    jpeg_warmup_steps: int = 300
    real_jpeg_probability: float = 0.0
    validate_every: int = 1
    max_train_steps: int | None = None
    amp: bool = True
    gates_only: bool = False
    router_only: bool = False

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.discriminator_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("Adam beta values must be within [0, 1)")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive")
        if not 1 <= self.jpeg_quality_min <= self.jpeg_quality_max <= 100:
            raise ValueError("JPEG quality range must be within [1, 100]")
        if self.validate_every < 1:
            raise ValueError("validate_every must be positive")
        if self.jpeg_warmup_steps < 0:
            raise ValueError("jpeg_warmup_steps cannot be negative")
        if not 0.0 <= self.real_jpeg_probability <= 1.0:
            raise ValueError("real_jpeg_probability must be in [0, 1]")
        if self.max_train_steps is not None and self.max_train_steps < 1:
            raise ValueError("max_train_steps must be positive or null")


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()
        self.loss.validate()
        self.train.validate()
        if self.model.architecture == "invertible":
            if self.model.payload_bits is not None:
                raise ValueError("invertible architecture supports full-image payloads only")
            if self.model.secret_size != self.data.image_size:
                raise ValueError(
                    "invertible architecture requires secret_size to match data.image_size"
                )
            if self.train.gates_only and self.model.invertible_blocks != 16:
                raise ValueError("gates-only warm-start adaptation requires 16 invertible blocks")
            if self.train.router_only and not self.model.orthogonal_router:
                raise ValueError("router-only adaptation requires orthogonal_router")
            adaptation_modes = (
                self.train.gates_only,
                self.train.router_only,
            )
            if sum(adaptation_modes) > 1:
                raise ValueError("gates_only and router_only are mutually exclusive")
        elif self.train.gates_only or self.train.router_only:
            raise ValueError("parameter-efficient adaptation modes require invertible architecture")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        if not isinstance(raw, dict):
            raise ValueError("experiment configuration must be an object")
        config = cls(
            model=ModelConfig(**_drop_removed_noops("model", raw.get("model", {}))),
            data=DataConfig(**_drop_removed_noops("data", raw.get("data", {}))),
            loss=LossConfig(**_drop_removed_noops("loss", raw.get("loss", {}))),
            train=TrainConfig(**_drop_removed_noops("train", raw.get("train", {}))),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)
