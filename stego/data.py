"""Deterministic, leakage-resistant image pairing for steganography."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def discover_images(roots: Sequence[str | Path]) -> list[Path]:
    images: list[Path] = []
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"image root does not exist: {root}")
        images.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    unique = sorted(set(images))
    if len(unique) < 2:
        raise RuntimeError("at least two images are required to form cover/secret pairs")
    return unique


def split_images(
    images: Sequence[Path], train_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    if len(images) < 4:
        raise ValueError("at least four images are required for a disjoint train/validation split")
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    boundary = max(2, min(len(shuffled) - 2, round(len(shuffled) * train_fraction)))
    return shuffled[:boundary], shuffled[boundary:]


def _load_image(path: Path, image_size: int, rng: random.Random, augment: bool) -> Tensor:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if augment:
            scale = rng.uniform(1.0, 1.2)
            target = max(image_size, round(image_size * scale))
            image = ImageOps.fit(image, (target, target), method=Image.Resampling.BICUBIC)
            maximum_offset = target - image_size
            left = rng.randint(0, maximum_offset) if maximum_offset else 0
            top = rng.randint(0, maximum_offset) if maximum_offset else 0
            image = image.crop((left, top, left + image_size, top + image_size))
            if rng.random() < 0.5:
                image = ImageOps.mirror(image)
        else:
            image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class ImagePairDataset(Dataset[tuple[Tensor, Tensor]]):
    """Pairs independent images without crossing train/validation partitions."""

    def __init__(
        self,
        images: Sequence[Path],
        image_size: int,
        *,
        augment: bool,
        seed: int,
        payload_bits: int | None = None,
        secret_image_size: int | None = None,
    ) -> None:
        if len(images) < 2:
            raise ValueError("at least two images are required")
        self.images = list(images)
        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        self.payload_bits = payload_bits
        self.secret_image_size = secret_image_size or image_size
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.images)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        rng = random.Random(self.seed + self.epoch * len(self.images) + index)
        cover = _load_image(self.images[index], self.image_size, rng, self.augment)
        if self.payload_bits is not None:
            payload = torch.tensor(
                [rng.getrandbits(1) for _ in range(self.payload_bits)],
                dtype=torch.float32,
            )
            return cover, payload
        secret_index = rng.randrange(len(self.images) - 1)
        if secret_index >= index:
            secret_index += 1
        secret = _load_image(
            self.images[secret_index],
            self.secret_image_size,
            rng,
            self.augment,
        )
        return cover, secret
