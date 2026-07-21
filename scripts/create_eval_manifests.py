#!/usr/bin/env python3
"""Create fresh, content-addressed development and final evaluation cohorts.

The current benchmark first calls ``discover_images`` and then samples 2,000
images with ``torch.randperm`` seeded with 2026. This script reconstructs that
sample exactly, excludes it, and builds image-disjoint cover/secret pairs from
the remaining files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from stego.data import discover_images

SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _root_labels(roots: list[Path]) -> dict[Path, str]:
    labels = [root.name for root in roots]
    if len(labels) != len(set(labels)):
        raise ValueError("data directory basenames must be unique for portable manifest paths")
    return dict(zip(roots, labels, strict=True))


def _portable_path(path: Path, roots: list[Path], labels: dict[Path, str]) -> str:
    for root in roots:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        return f"{labels[root]}/{relative.as_posix()}"
    raise ValueError(f"image is outside the declared data roots: {path}")


def _sample_indices(size: int, count: int, seed: int) -> list[int]:
    if not 0 <= count <= size:
        raise ValueError(f"sample count must be within [0, {size}], got {count}")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(size, generator=generator)[:count].tolist()


def _pair_records(
    cohort: str,
    images: list[Path],
    pairs: int,
    roots: list[Path],
    labels: dict[Path, str],
) -> list[dict[str, str]]:
    expected = pairs * 2
    if len(images) != expected:
        raise ValueError(f"{cohort} requires {expected} unique images, got {len(images)}")
    covers = images[:pairs]
    secrets = images[pairs:]
    records: list[dict[str, str]] = []
    for index, (cover, secret) in enumerate(zip(covers, secrets, strict=True)):
        records.append(
            {
                "pair_id": f"{cohort}-{index:04d}",
                "cover_path": _portable_path(cover, roots, labels),
                "cover_sha256": _sha256_file(cover),
                "secret_path": _portable_path(secret, roots, labels),
                "secret_sha256": _sha256_file(secret),
            }
        )
    return records


def _manifest(
    *,
    cohort: str,
    records: list[dict[str, str]],
    root_references: dict[str, str],
    discovered_images: int,
    excluded_images: int,
    benchmark_seed: int,
    cohort_seed: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cohort": cohort,
        "data_roots": root_references,
        "path_resolution": "<data-root label>/<path relative to that root>",
        "sampling": {
            "discovered_images": discovered_images,
            "excluded_benchmark_images": excluded_images,
            "benchmark_sample_seed": benchmark_seed,
            "cohort_sample_seed": cohort_seed,
            "pair_count": len(records),
            "unique_image_count": len(records) * 2,
            "all_roles_and_cohorts_image_disjoint": True,
        },
        "pairs": records,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{_sha256_file(path)}  {path.name}\n", encoding="utf-8")


def create_manifests(
    data_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    excluded_images: int = 2000,
    dev_pairs: int = 100,
    final_pairs: int = 200,
    benchmark_seed: int = 2026,
    cohort_seed: int = 2027,
) -> tuple[Path, Path]:
    """Create deterministic manifests and return their paths."""

    if dev_pairs < 1 or final_pairs < 1:
        raise ValueError("dev_pairs and final_pairs must be positive")
    roots = [Path(value).expanduser().resolve() for value in data_dirs]
    labels = _root_labels(roots)
    images = discover_images(roots)
    destination = Path(output_dir).expanduser().resolve()
    root_references = {
        labels[root]: Path(os.path.relpath(root, destination)).as_posix() for root in roots
    }

    benchmark_indices = _sample_indices(len(images), excluded_images, benchmark_seed)
    benchmark_paths = {images[index] for index in benchmark_indices}
    unused = [path for path in images if path not in benchmark_paths]

    required_images = 2 * (dev_pairs + final_pairs)
    if len(unused) < required_images:
        raise ValueError(
            f"need {required_images} unused images for the requested cohorts, "
            f"but only {len(unused)} remain after excluding {excluded_images}"
        )

    selected_indices = _sample_indices(len(unused), required_images, cohort_seed)
    selected = [unused[index] for index in selected_indices]
    dev_boundary = dev_pairs * 2
    dev_images = selected[:dev_boundary]
    final_images = selected[dev_boundary:]
    if set(dev_images) & set(final_images):
        raise RuntimeError("internal error: development and final cohorts overlap")

    dev = _manifest(
        cohort="dev",
        records=_pair_records("dev", dev_images, dev_pairs, roots, labels),
        root_references=root_references,
        discovered_images=len(images),
        excluded_images=excluded_images,
        benchmark_seed=benchmark_seed,
        cohort_seed=cohort_seed,
    )
    final = _manifest(
        cohort="final",
        records=_pair_records("final", final_images, final_pairs, roots, labels),
        root_references=root_references,
        discovered_images=len(images),
        excluded_images=excluded_images,
        benchmark_seed=benchmark_seed,
        cohort_seed=cohort_seed,
    )

    dev_path = destination / "dev.json"
    final_path = destination / "final.json"
    _write_manifest(dev_path, dev)
    _write_manifest(final_path, final)
    return dev_path, final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/eval_manifests")
    parser.add_argument("--excluded-images", type=int, default=2000)
    parser.add_argument("--dev-pairs", type=int, default=100)
    parser.add_argument("--final-pairs", type=int, default=200)
    parser.add_argument("--benchmark-seed", type=int, default=2026)
    parser.add_argument("--cohort-seed", type=int, default=2027)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dev_path, final_path = create_manifests(
        args.data_dir,
        args.output_dir,
        excluded_images=args.excluded_images,
        dev_pairs=args.dev_pairs,
        final_pairs=args.final_pairs,
        benchmark_seed=args.benchmark_seed,
        cohort_seed=args.cohort_seed,
    )
    for path in (dev_path, final_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(f"{payload['cohort']}: {path} sha256={payload['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
