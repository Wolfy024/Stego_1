"""Adversarial training, validation, and checkpoint utilities."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from stego.config import ExperimentConfig
from stego.data import ImagePairDataset
from stego.jpeg import DifferentiableJPEG
from stego.metrics import bit_accuracy, image_metrics, real_jpeg_roundtrip
from stego.models import StegoGAN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


@contextmanager
def frozen(module: nn.Module) -> Iterator[None]:
    parameters = list(module.parameters())
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, previous, strict=True):
            parameter.requires_grad_(requires_grad)


def total_variation(image: Tensor) -> Tensor:
    horizontal = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()
    vertical = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean()
    return horizontal + vertical


class MetricAccumulator:
    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def update(self, values: dict[str, Tensor | float], count: int | None = None) -> None:
        for name, value in values.items():
            if isinstance(value, Tensor):
                finite = value.detach().float().flatten()
                finite = finite[torch.isfinite(finite)]
                if finite.numel() == 0:
                    continue
                self.totals[name] += finite.sum().item()
                self.counts[name] += finite.numel()
            else:
                item_count = 1 if count is None else count
                if math.isfinite(value):
                    self.totals[name] += value * item_count
                    self.counts[name] += item_count

    def means(self) -> dict[str, float]:
        return {
            name: self.totals[name] / self.counts[name]
            for name in sorted(self.totals)
            if self.counts[name]
        }


@dataclass(slots=True)
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    nonfinite_steps: int = 0


class Trainer:
    def __init__(
        self,
        model: StegoGAN,
        config: ExperimentConfig,
        output_dir: str | Path,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        if config.train.gates_only:
            for name, parameter in model.encoder.named_parameters():
                parameter.requires_grad_(".attention." in name or ".band_gate." in name)
        elif config.train.router_only:
            for name, parameter in model.encoder.named_parameters():
                parameter.requires_grad_(".router." in name)
        generator_parameters = model.generator_parameters()
        self.generator_optimizer = Adam(
            generator_parameters,
            lr=config.train.learning_rate,
            betas=(config.train.beta1, config.train.beta2),
        )
        self.discriminator_optimizer = Adam(
            model.discriminator.parameters(),
            lr=config.train.discriminator_learning_rate,
            betas=(config.train.beta1, config.train.beta2),
        )
        amp_enabled = config.train.amp and device.type == "cuda"
        self.amp_enabled = amp_enabled
        self.amp_dtype = (
            torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
        )
        scale_enabled = amp_enabled and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler(device.type, enabled=scale_enabled)
        self.jpeg = DifferentiableJPEG(differentiable=True).to(device)
        self.state = TrainingState()
        self.history_path = self.output_dir / "history.csv"
        config.save(self.output_dir / "config.json")

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        )

    def _step_discriminator(self, cover: Tensor, stego: Tensor) -> Tensor:
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            real_logits = self.model.discriminator(cover)
            fake_logits = self.model.discriminator(stego.detach())
            loss = 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())
        self.scaler.scale(loss).backward()
        self.scaler.step(self.discriminator_optimizer)
        return loss.detach()

    def _step_generator(
        self, cover: Tensor, secret: Tensor, outputs: dict[str, Tensor], jpeg_quality: int
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self.generator_optimizer.zero_grad(set_to_none=True)
        stego = outputs["stego"]
        revealed = outputs["revealed_secret"]
        with self._autocast(), frozen(self.model.discriminator):
            invertible = self.config.model.architecture == "invertible"
            cover_loss = F.mse_loss(stego, cover) if invertible else F.smooth_l1_loss(stego, cover)
            warmup = self.config.train.jpeg_warmup_steps
            if warmup == 0:
                robust_scale = 1.0
            else:
                robust_scale = min(
                    1.0,
                    max(0.0, (self.state.global_step - warmup) / warmup),
                )
            if robust_scale > 0:
                if random.random() < self.config.train.real_jpeg_probability:
                    real_jpeg = real_jpeg_roundtrip(stego, quality=jpeg_quality)
                    jpeg_stego = stego + (real_jpeg - stego).detach()
                else:
                    jpeg_stego = self.jpeg(stego, quality=jpeg_quality)
                robust_revealed = self.model.reveal(jpeg_stego)
            else:
                robust_revealed = None
            if secret.ndim == 2:
                secret_loss = F.binary_cross_entropy_with_logits(revealed, secret)
                robust_loss = (
                    F.binary_cross_entropy_with_logits(robust_revealed, secret)
                    if robust_revealed is not None
                    else stego.new_zeros(())
                )
            else:
                secret_loss = (
                    F.mse_loss(revealed, secret) if invertible else F.l1_loss(revealed, secret)
                )
                robust_loss = (
                    (
                        F.mse_loss(robust_revealed, secret)
                        if invertible
                        else F.l1_loss(robust_revealed, secret)
                    )
                    if robust_revealed is not None
                    else stego.new_zeros(())
                )
            adversarial_loss = (
                -self.model.discriminator(stego).mean()
                if self.config.loss.adversarial > 0
                else stego.new_zeros(())
            )
            variation_loss = total_variation(outputs["residual"])
            frequency_loss = (
                F.mse_loss(outputs["stego_low"], outputs["cover_low"])
                if "stego_low" in outputs
                else stego.new_zeros(())
            )
            weights = self.config.loss
            loss = (
                weights.cover * cover_loss
                + weights.secret * secret_loss
                + weights.robust_secret * robust_scale * robust_loss
                + weights.adversarial * adversarial_loss
                + weights.residual_tv * variation_loss
                + weights.low_frequency * frequency_loss
            )
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.generator_optimizer)
        parameters = self.model.generator_parameters()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.train.grad_clip)
        if torch.isfinite(loss) and torch.isfinite(grad_norm):
            self.scaler.step(self.generator_optimizer)
        else:
            self.state.nonfinite_steps += 1
            self.generator_optimizer.zero_grad(set_to_none=True)
        self.scaler.update()
        components = {
            "train_gen_loss": loss.detach(),
            "train_cover_loss": cover_loss.detach(),
            "train_secret_loss": secret_loss.detach(),
            "train_robust_loss": robust_loss.detach(),
            "train_robust_weight": stego.new_tensor(weights.robust_secret * robust_scale),
            "train_adversarial_loss": adversarial_loss.detach(),
            "train_grad_norm": grad_norm.detach(),
            "train_frequency_loss": frequency_loss.detach(),
        }
        return loss.detach(), components

    def train_epoch(self, loader: DataLoader[tuple[Tensor, Tensor]]) -> dict[str, float]:
        self.model.train()
        dataset = loader.dataset
        if isinstance(dataset, ImagePairDataset):
            dataset.set_epoch(self.state.epoch)
        metrics = MetricAccumulator()
        started = time.monotonic()
        processed_images = 0
        for batch_index, (cover, secret) in enumerate(loader):
            if (
                self.config.train.max_train_steps is not None
                and self.state.global_step >= self.config.train.max_train_steps
            ):
                break
            cover = cover.to(self.device, non_blocking=True)
            secret = secret.to(self.device, non_blocking=True)
            processed_images += cover.shape[0]
            with self._autocast():
                outputs = self.model(cover, secret)
            discriminator_loss = (
                self._step_discriminator(cover, outputs["stego"])
                if self.config.loss.adversarial > 0
                else cover.new_zeros(())
            )
            quality = random.randint(
                self.config.train.jpeg_quality_min, self.config.train.jpeg_quality_max
            )
            _, components = self._step_generator(cover, secret, outputs, quality)
            components["train_discriminator_loss"] = discriminator_loss
            components["train_jpeg_quality"] = float(quality)
            metrics.update(components)
            self.state.global_step += 1
            if batch_index % 25 == 0:
                print(
                    f"epoch={self.state.epoch + 1} step={self.state.global_step} "
                    f"g={components['train_gen_loss'].item():.4f} "
                    f"d={discriminator_loss.item():.4f} q={quality}",
                    flush=True,
                )
        elapsed = max(time.monotonic() - started, 1e-9)
        result = metrics.means()
        result["train_images_per_second"] = processed_images / elapsed
        result["nonfinite_steps"] = float(self.state.nonfinite_steps)
        return result

    @torch.inference_mode()
    def evaluate(
        self,
        loader: DataLoader[tuple[Tensor, Tensor]],
        *,
        jpeg_qualities: tuple[int, ...] = (50, 70, 90),
        max_batches: int | None = None,
        real_codec: bool = True,
    ) -> dict[str, float]:
        self.model.eval()
        metrics = MetricAccumulator()
        evaluated_images = 0
        for batch_index, (cover, secret) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            cover = cover.to(self.device, non_blocking=True)
            secret = secret.to(self.device, non_blocking=True)
            evaluated_images += cover.shape[0]
            outputs = self.model(cover.float(), secret.float())
            stego = outputs["stego"].float()
            revealed = outputs["revealed_secret"].float()
            metrics.update(image_metrics(cover.float(), stego, "cover"))
            if secret.ndim == 2:
                metrics.update({"secret_accuracy_pct": bit_accuracy(secret, revealed)})
            else:
                metrics.update(image_metrics(secret.float(), revealed, "secret"))
            for quality in jpeg_qualities:
                degraded = (
                    real_jpeg_roundtrip(stego, quality)
                    if real_codec
                    else self.jpeg(stego, quality=quality)
                )
                robust_revealed = self.model.reveal(degraded).float()
                if secret.ndim == 2:
                    metrics.update(
                        {f"jpeg_q{quality}_accuracy_pct": bit_accuracy(secret, robust_revealed)}
                    )
                else:
                    metrics.update(
                        image_metrics(secret.float(), robust_revealed, f"jpeg_q{quality}")
                    )
        result = metrics.means()
        result["evaluated_images"] = float(evaluated_images)
        return result

    def save_checkpoint(self, name: str = "latest.pt") -> Path:
        destination = self.output_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        payload = {
            "config": self.config.to_dict(),
            "model": self.model.state_dict(),
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "nonfinite_steps": self.state.nonfinite_steps,
            },
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        return destination

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        resume_optimizers: bool = True,
        resume_state: bool = True,
        allow_new_adapters: bool = False,
    ) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if allow_new_adapters:
            source_model = payload["config"].get("model", {})
            allow_new_router = self.config.model.orthogonal_router and not source_model.get(
                "orthogonal_router", False
            )
            incompatible = self.model.load_state_dict(payload["model"], strict=False)
            invalid_missing = [
                key
                for key in incompatible.missing_keys
                if not (
                    allow_new_router
                    and key.startswith("encoder.flow.blocks.")
                    and ".router." in key
                )
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "incompatible warm start: "
                    f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
                )
        else:
            self.model.load_state_dict(payload["model"])
        if resume_optimizers:
            self.generator_optimizer.load_state_dict(payload["generator_optimizer"])
            self.discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
            self.scaler.load_state_dict(payload["scaler"])
        if resume_state:
            self.state = TrainingState(**payload["state"])

    def _append_history(self, row: dict[str, float]) -> None:
        row = {"epoch": self.state.epoch + 1, "global_step": self.state.global_step, **row}
        write_header = not self.history_path.exists()
        with self.history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def fit(
        self,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        validation_loader: DataLoader[tuple[Tensor, Tensor]],
    ) -> Path:
        for epoch in range(self.state.epoch, self.config.train.epochs):
            self.state.epoch = epoch
            train_metrics = self.train_epoch(train_loader)
            validation_metrics: dict[str, float] = {}
            if (epoch + 1) % self.config.train.validate_every == 0:
                validation_metrics = self.evaluate(validation_loader, real_codec=False)
            row = {**train_metrics, **validation_metrics}
            self._append_history(row)
            checkpoint = self.save_checkpoint("latest.pt")
            summary = {key: round(value, 4) for key, value in row.items()}
            print(f"epoch={epoch + 1} metrics={json.dumps(summary, sort_keys=True)}", flush=True)
            if (
                self.config.train.max_train_steps is not None
                and self.state.global_step >= self.config.train.max_train_steps
            ):
                return checkpoint
        return self.output_dir / "latest.pt"


def load_model_from_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[StegoGAN, ExperimentConfig]:
    payload = torch.load(path, map_location=device, weights_only=True)
    config = ExperimentConfig.from_dict(payload["config"])
    model = StegoGAN(config.model).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config
