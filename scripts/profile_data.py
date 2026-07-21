"""Validate image corpora and save a compact, reproducible quality profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image

from stego.data import discover_images, split_images


def profile(roots: list[Path], train_fraction: float, seed: int) -> dict[str, object]:
    images = discover_images(roots)
    source_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    hashes: Counter[str] = Counter()
    unreadable: list[str] = []
    widths: list[int] = []
    heights: list[int] = []

    resolved_roots = [root.resolve() for root in roots]
    for path in images:
        source = next(
            (root.name for root in resolved_roots if root == path or root in path.parents),
            "unknown",
        )
        source_counts[source] += 1
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[digest.hexdigest()] += 1
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                widths.append(width)
                heights.append(height)
                format_counts[str(image.format)] += 1
                dimension_counts[f"{width}x{height}"] += 1
        except (OSError, ValueError) as error:
            unreadable.append(f"{path}: {error}")

    train, validation = split_images(images, train_fraction, seed)
    duplicate_groups = sum(count > 1 for count in hashes.values())
    duplicate_files = sum(count - 1 for count in hashes.values() if count > 1)
    return {
        "schema_version": 1,
        "grain": "one decoded RGB image file",
        "roots": [root.as_posix() for root in roots],
        "images_total": len(images),
        "source_counts": dict(sorted(source_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "unique_dimensions": len(dimension_counts),
        "width_range_px": [min(widths), max(widths)] if widths else None,
        "height_range_px": [min(heights), max(heights)] if heights else None,
        "unreadable_count": len(unreadable),
        "unreadable_examples": unreadable[:10],
        "exact_duplicate_groups": duplicate_groups,
        "exact_duplicate_files": duplicate_files,
        "train_images": len(train),
        "validation_images": len(validation),
        "split_overlap": len(set(train) & set(validation)),
        "split_seed": seed,
        "train_fraction": train_fraction,
        "checks": {
            "all_images_readable": not unreadable,
            "no_exact_duplicates": duplicate_files == 0,
            "split_is_disjoint": not (set(train) & set(validation)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", nargs="+", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/data_profile.json"))
    args = parser.parse_args()
    result = profile(args.data_dir, args.train_fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
