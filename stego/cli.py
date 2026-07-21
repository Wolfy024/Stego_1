"""Command-line interface for smoke tests, training, and evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stego.config import ExperimentConfig, ModelConfig
from stego.data import ImagePairDataset, discover_images, split_images
from stego.jpeg import DifferentiableJPEG
from stego.metrics import bit_accuracy, image_metrics
from stego.models import StegoGAN
from stego.training import Trainer, load_model_from_checkpoint, seed_everything, select_device


def _loader(
    dataset: ImagePairDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=shuffle and len(dataset) >= batch_size,
    )


def _datasets(
    roots: list[str],
    config: ExperimentConfig,
) -> tuple[ImagePairDataset, ImagePairDataset, int, int]:
    images = discover_images(roots)
    discovered_images = len(images)
    if config.data.max_images is not None and len(images) > config.data.max_images:
        generator = torch.Generator().manual_seed(config.data.seed)
        order = torch.randperm(len(images), generator=generator).tolist()
        images = [images[index] for index in order[: config.data.max_images]]
    train_images, validation_images = split_images(
        images, config.data.train_fraction, config.data.seed
    )
    train_dataset = ImagePairDataset(
        train_images,
        config.data.image_size,
        augment=True,
        seed=config.data.seed,
        payload_bits=config.model.payload_bits,
        secret_image_size=config.model.secret_size,
    )
    validation_dataset = ImagePairDataset(
        validation_images,
        config.data.image_size,
        augment=False,
        seed=config.data.seed + 1,
        payload_bits=config.model.payload_bits,
        secret_image_size=config.model.secret_size,
    )
    return train_dataset, validation_dataset, len(images), discovered_images


def _source_counts(dataset: ImagePairDataset, roots: list[str]) -> dict[str, int]:
    resolved_roots = [Path(root).resolve() for root in roots]
    counts = {root.name: 0 for root in resolved_roots}
    for image in dataset.images:
        for root in resolved_roots:
            if root == image or root in image.parents:
                counts[root.name] += 1
                break
    return counts


def _apply_train_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    mappings = {
        "epochs": (config.train, "epochs"),
        "batch_size": (config.train, "batch_size"),
        "max_steps": (config.train, "max_train_steps"),
        "image_size": (config.data, "image_size"),
        "max_images": (config.data, "max_images"),
        "workers": (config.data, "workers"),
        "base_channels": (config.model, "base_channels"),
        "payload_bits": (config.model, "payload_bits"),
        "max_residual": (config.model, "max_residual"),
        "target_cover_psnr": (config.model, "target_cover_psnr"),
        "secret_size": (config.model, "secret_size"),
        "validate_every": (config.train, "validate_every"),
        "jpeg_quality_min": (config.train, "jpeg_quality_min"),
        "jpeg_quality_max": (config.train, "jpeg_quality_max"),
        "real_jpeg_probability": (config.train, "real_jpeg_probability"),
        "jpeg_warmup_steps": (config.train, "jpeg_warmup_steps"),
        "cover_weight": (config.loss, "cover"),
        "secret_weight": (config.loss, "secret"),
        "low_frequency_weight": (config.loss, "low_frequency"),
    }
    for argument, (target, attribute) in mappings.items():
        value = getattr(args, argument, None)
        if value is not None:
            setattr(target, attribute, value)
    if getattr(args, "gates_only", False):
        config.train.gates_only = True
    config.validate()


def run_smoke(args: argparse.Namespace) -> int:
    device = select_device(args.device)
    config = ModelConfig(
        architecture=args.architecture,
        base_channels=args.base_channels,
        max_residual=args.max_residual,
        target_cover_psnr=args.target_cover_psnr,
        spectral_norm=False,
        payload_bits=args.payload_bits,
        secret_size=args.secret_size,
        invertible_blocks=args.invertible_blocks,
        coupling_channels=args.coupling_channels,
        coupling_clamp=args.coupling_clamp,
        latent_noise_std=args.latent_noise_std,
    )
    model = StegoGAN(config).to(device).train()
    cover = torch.rand(args.batch_size, 3, args.image_size, args.image_size, device=device) * 2 - 1
    if args.payload_bits is None:
        secret = (
            torch.rand(
                args.batch_size,
                3,
                args.secret_size,
                args.secret_size,
                device=device,
            )
            * 2
            - 1
        )
    else:
        secret = torch.randint(
            0,
            2,
            (args.batch_size, args.payload_bits),
            device=device,
            dtype=torch.float32,
        )
    outputs = model(cover, secret)
    jpeg = DifferentiableJPEG(quality=50).to(device)
    restored = model.reveal(jpeg(outputs["stego"]))
    loss = (
        torch.nn.functional.l1_loss(restored, secret)
        if args.payload_bits is None
        else torch.nn.functional.binary_cross_entropy_with_logits(restored, secret)
    )
    loss.backward()
    metrics = {
        name: value.mean().item()
        for name, value in {
            **image_metrics(cover, outputs["stego"], "cover"),
            **(
                image_metrics(secret, outputs["revealed_secret"], "secret")
                if args.payload_bits is None
                else {
                    "secret_accuracy_pct": bit_accuracy(secret, outputs["revealed_secret"]),
                    "jpeg_q50_accuracy_pct": bit_accuracy(secret, restored),
                }
            ),
        }.items()
    }
    summary = {
        "device": str(device),
        "architecture": config.architecture,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "bottleneck_channels": model.bottleneck_channels,
        "stego_shape": list(outputs["stego"].shape),
        "jpeg_gradient_finite": all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.encoder.parameters()
        ),
        **metrics,
    }
    print(json.dumps(summary, indent=2))
    return 0


def run_train(args: argparse.Namespace) -> int:
    device = select_device(args.device)
    if args.resume:
        _, config = load_model_from_checkpoint(args.resume, torch.device("cpu"))
    else:
        config = ExperimentConfig.load(args.config)
    _apply_train_overrides(config, args)
    seed_everything(config.data.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    train_dataset, validation_dataset, sampled_images, discovered_images = _datasets(
        args.data_dir, config
    )
    train_loader = _loader(
        train_dataset,
        config.train.batch_size,
        config.data.workers,
        shuffle=True,
        device=device,
    )
    validation_loader = _loader(
        validation_dataset,
        config.train.batch_size,
        config.data.workers,
        shuffle=False,
        device=device,
    )
    model = StegoGAN(config.model)
    trainer = Trainer(model, config, args.output_dir, device)
    if args.resume:
        trainer.load_checkpoint(args.resume, resume_optimizers=not config.train.gates_only)
    print(
        json.dumps(
            {
                "device": str(device),
                "images_discovered": discovered_images,
                "images_sampled": sampled_images,
                "images_train": len(train_dataset),
                "images_validation": len(validation_dataset),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.generator_parameters()
                ),
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    checkpoint = trainer.fit(train_loader, validation_loader)
    print(f"checkpoint={checkpoint.resolve()}")
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    device = select_device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    config.data.workers = args.workers
    train_dataset, validation_dataset, sampled_images, discovered_images = _datasets(
        args.data_dir, config
    )
    loader = _loader(
        validation_dataset,
        args.batch_size or config.train.batch_size,
        args.workers,
        shuffle=False,
        device=device,
    )
    output_path = Path(args.output)
    trainer = Trainer(model, config, output_path.parent / ".evaluation", device)
    results = trainer.evaluate(
        loader,
        jpeg_qualities=tuple(args.jpeg_qualities),
        max_batches=args.max_batches,
        real_codec=True,
    )
    payload = {
        "schema_version": 2,
        "checkpoint": Path(args.checkpoint).as_posix(),
        "data_roots": [Path(root).as_posix() for root in args.data_dir],
        "payload_type": "bit watermark" if config.model.payload_bits else "RGB secret image",
        "cover_size_px": [config.data.image_size, config.data.image_size],
        "secret_size_px": (
            None
            if config.model.payload_bits
            else [config.model.secret_size, config.model.secret_size]
        ),
        "dataset_images_discovered": discovered_images,
        "dataset_images_sampled": sampled_images,
        "train_images": len(train_dataset),
        "validation_images_available": len(validation_dataset),
        "train_source_counts": _source_counts(train_dataset, args.data_dir),
        "validation_source_counts": _source_counts(validation_dataset, args.data_dir),
        "split_seed": config.data.seed,
        "train_fraction": config.data.train_fraction,
        "jpeg_codec": "Pillow/libjpeg, 4:2:0 subsampling",
        "metrics": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run forward/backward and JPEG gradient checks")
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--architecture", choices=("unet", "invertible"), default="unet")
    smoke.add_argument("--base-channels", type=int, default=8)
    smoke.add_argument("--max-residual", type=float, default=0.03)
    smoke.add_argument("--target-cover-psnr", type=float, default=38.5)
    smoke.add_argument("--image-size", type=int, default=32)
    smoke.add_argument("--batch-size", type=int, default=1)
    smoke.add_argument("--payload-bits", type=int)
    smoke.add_argument("--secret-size", type=int, default=32)
    smoke.add_argument("--invertible-blocks", type=int, default=1)
    smoke.add_argument("--coupling-channels", type=int, default=8)
    smoke.add_argument("--coupling-clamp", type=float, default=2.0)
    smoke.add_argument("--latent-noise-std", type=float, default=1.0)
    smoke.set_defaults(function=run_smoke)

    train = subparsers.add_parser("train", help="train on one or more image directories")
    train.add_argument("--data-dir", nargs="+", required=True)
    train.add_argument("--config", default="configs/image_h100.json")
    train.add_argument("--output-dir", default="runs/experiment")
    train.add_argument("--resume")
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--image-size", type=int)
    train.add_argument("--max-images", type=int)
    train.add_argument("--workers", type=int)
    train.add_argument("--base-channels", type=int)
    train.add_argument("--payload-bits", type=int)
    train.add_argument("--max-residual", type=float)
    train.add_argument("--target-cover-psnr", type=float)
    train.add_argument("--secret-size", type=int)
    train.add_argument("--validate-every", type=int)
    train.add_argument("--jpeg-quality-min", type=int)
    train.add_argument("--jpeg-quality-max", type=int)
    train.add_argument("--real-jpeg-probability", type=float)
    train.add_argument("--jpeg-warmup-steps", type=int)
    train.add_argument("--cover-weight", type=float)
    train.add_argument("--secret-weight", type=float)
    train.add_argument("--low-frequency-weight", type=float)
    train.add_argument("--gates-only", action="store_true")
    train.set_defaults(function=run_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint with real JPEG")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data-dir", nargs="+", required=True)
    evaluate.add_argument("--output", default="results/benchmark.json")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--workers", type=int, default=2)
    evaluate.add_argument("--max-batches", type=int)
    evaluate.add_argument("--jpeg-qualities", type=int, nargs="+", default=[50, 70, 90])
    evaluate.set_defaults(function=run_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
