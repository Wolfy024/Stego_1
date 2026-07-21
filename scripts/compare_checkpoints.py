"""Compare two steganography checkpoints on an explicit paired-image manifest.

Manifest format::

    {
      "data_roots": {"coco": "../../data/coco/val2017"},
      "pairs": [{
        "pair_id": "pair-001",
        "cover_path": "coco/001.png",
        "cover_sha256": "...",
        "secret_path": "coco/002.png",
        "secret_sha256": "..."
      }]
    }

The compact ``id``/``cover``/``secret`` form is also supported, with relative
paths resolved from the manifest directory. The candidate is always compared
against the baseline, so every reported delta is candidate minus baseline and
positive image-quality deltas indicate improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor, nn

from stego.metrics import image_metrics
from stego.training import load_model_from_checkpoint, select_device

METRIC_NAMES = (
    "cover_psnr_db",
    "cover_ssim",
    "cover_similarity_pct",
    "secret_psnr_db",
    "secret_ssim",
    "secret_similarity_pct",
)


@dataclass(frozen=True, slots=True)
class PairSpec:
    """One explicit cover/secret comparison unit."""

    identifier: str
    cover_reference: str
    secret_reference: str
    cover_path: Path
    secret_path: Path
    cover_sha256: str | None = None
    secret_sha256: str | None = None


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _resolve_data_roots(raw: dict[str, Any], manifest_directory: Path) -> dict[str, Path]:
    encoded = raw.get("data_roots", {})
    if not isinstance(encoded, dict):
        raise ValueError("manifest data_roots must be an object mapping labels to paths")
    roots: dict[str, Path] = {}
    for label, reference in encoded.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("manifest data-root labels must be non-empty strings")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"manifest data root {label!r} must be a non-empty path")
        root = Path(reference).expanduser()
        if not root.is_absolute():
            root = manifest_directory / root
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"manifest data root does not exist: {root}")
        roots[label] = root
    return roots


def _resolve_image(
    reference: str,
    manifest_directory: Path,
    field: str,
    data_roots: dict[str, Path],
) -> Path:
    if not reference.strip():
        raise ValueError(f"pair {field} path cannot be empty")
    path = Path(reference).expanduser()
    declared_root: Path | None = None
    if not path.is_absolute() and path.parts and path.parts[0] in data_roots:
        declared_root = data_roots[path.parts[0]]
        path = declared_root.joinpath(*path.parts[1:])
    elif not path.is_absolute():
        path = manifest_directory / path
    path = path.resolve()
    if declared_root is not None:
        try:
            path.relative_to(declared_root)
        except ValueError as error:
            raise ValueError(f"pair {field} path escapes data root: {reference}") from error
    if not path.is_file():
        raise FileNotFoundError(f"pair {field} image does not exist: {path}")
    return path


def _optional_sha256(item: dict[str, Any], field: str, index: int) -> str | None:
    value = item.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"manifest pair {index} {field} must be a 64-character hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"manifest pair {index} {field} is not hexadecimal") from error
    return value.lower()


def load_manifest(path: str | Path) -> list[PairSpec]:
    """Load and validate a manifest, resolving image paths without reordering pairs."""

    manifest_path = Path(path).expanduser().resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("pairs"), list):
        raise ValueError("manifest must be a JSON object with a top-level 'pairs' list")
    if not raw["pairs"]:
        raise ValueError("manifest must contain at least one cover/secret pair")

    declared_manifest_sha256 = raw.get("manifest_sha256")
    if declared_manifest_sha256 is not None:
        if not isinstance(declared_manifest_sha256, str):
            raise ValueError("manifest_sha256 must be a string")
        unhashed = dict(raw)
        unhashed.pop("manifest_sha256")
        actual_manifest_sha256 = _canonical_sha256(unhashed)
        if declared_manifest_sha256 != actual_manifest_sha256:
            raise ValueError(
                "manifest_sha256 mismatch: "
                f"declared {declared_manifest_sha256}, computed {actual_manifest_sha256}"
            )

    manifest_directory = manifest_path.parent
    data_roots = _resolve_data_roots(raw, manifest_directory)
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[Path, Path]] = set()
    seen_images: set[Path] = set()
    pairs: list[PairSpec] = []
    for index, item in enumerate(raw["pairs"]):
        if not isinstance(item, dict):
            raise ValueError(f"manifest pair {index} must be a JSON object")
        cover_reference = item.get("cover_path", item.get("cover"))
        secret_reference = item.get("secret_path", item.get("secret"))
        if not isinstance(cover_reference, str) or not isinstance(secret_reference, str):
            raise ValueError(f"manifest pair {index} requires string cover and secret paths")
        identifier = item.get("pair_id", item.get("id", str(index)))
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"manifest pair {index} id must be a non-empty string")
        if identifier in seen_ids:
            raise ValueError(f"duplicate manifest pair id: {identifier}")

        cover_path = _resolve_image(cover_reference, manifest_directory, "cover", data_roots)
        secret_path = _resolve_image(secret_reference, manifest_directory, "secret", data_roots)
        signature = (cover_path, secret_path)
        if signature in seen_pairs:
            raise ValueError(
                f"duplicate cover/secret pair at manifest index {index}: "
                f"{cover_reference}, {secret_reference}"
            )
        if cover_path == secret_path:
            raise ValueError(f"manifest pair {index} uses the same image for both roles")
        reused = [path for path in signature if path in seen_images]
        if reused:
            raise ValueError(f"manifest image reused across pairs or roles: {reused[0]}")

        cover_sha256 = _optional_sha256(item, "cover_sha256", index)
        secret_sha256 = _optional_sha256(item, "secret_sha256", index)
        for label, image_path, expected in (
            ("cover", cover_path, cover_sha256),
            ("secret", secret_path, secret_sha256),
        ):
            if expected is not None:
                actual = _sha256(image_path)
                if actual != expected:
                    raise ValueError(
                        f"manifest pair {index} {label} hash mismatch: "
                        f"declared {expected}, computed {actual}"
                    )
        seen_ids.add(identifier)
        seen_pairs.add(signature)
        seen_images.update(signature)
        pairs.append(
            PairSpec(
                identifier=identifier,
                cover_reference=cover_reference,
                secret_reference=secret_reference,
                cover_path=cover_path,
                secret_path=secret_path,
                cover_sha256=cover_sha256,
                secret_sha256=secret_sha256,
            )
        )
    return pairs


def load_rgb_image(path: Path, image_size: int) -> Tensor:
    """Load one image with the same center-fit and [-1, 1] range used for validation."""

    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = ImageOps.fit(
                image,
                (image_size, image_size),
                method=Image.Resampling.BICUBIC,
            )
            array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    except (OSError, ValueError) as error:
        raise ValueError(f"could not decode RGB image {path}: {error}") from error
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def paired_bootstrap_mean_ci(
    deltas: np.ndarray | list[float],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Return a seeded percentile bootstrap CI for a paired mean delta."""

    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired bootstrap requires a non-empty one-dimensional delta array")
    if not np.isfinite(values).all():
        raise ValueError("paired bootstrap deltas must all be finite")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    if seed < 0:
        raise ValueError("bootstrap seed cannot be negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be within (0, 1)")

    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    rows_per_chunk = max(1, min(samples, 2_000_000 // values.size))
    for start in range(0, samples, rows_per_chunk):
        stop = min(samples, start + rows_per_chunk)
        indices = generator.integers(0, values.size, size=(stop - start, values.size))
        bootstrap_means[start:stop] = values[indices].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, (tail, 1.0 - tail))
    return {
        "method": "paired percentile bootstrap",
        "confidence_level": confidence,
        "samples": samples,
        "seed": seed,
        "lower": float(lower),
        "upper": float(upper),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def summarize_records(
    records: list[dict[str, Any]], *, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, Any]:
    """Aggregate paired records without discarding their shared comparison grain."""

    if not records:
        raise ValueError("at least one paired comparison record is required")
    baseline: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    candidate: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    deltas: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}

    for index, record in enumerate(records):
        for name in METRIC_NAMES:
            try:
                baseline_value = float(record["baseline_metrics"][name])
                candidate_value = float(record["candidate_metrics"][name])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"record {index} is missing numeric metric {name}") from error
            if not np.isfinite((baseline_value, candidate_value)).all():
                raise ValueError(f"record {index} contains non-finite metric {name}")
            delta = candidate_value - baseline_value
            baseline[name].append(baseline_value)
            candidate[name].append(candidate_value)
            deltas[name].append(delta)

    aggregate = {
        "baseline_mean": {name: _mean(values) for name, values in baseline.items()},
        "candidate_mean": {name: _mean(values) for name, values in candidate.items()},
        "candidate_minus_baseline": {
            name: {
                "mean": _mean(values),
                "median": float(np.median(np.asarray(values, dtype=np.float64))),
                "sample_std": (
                    float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
                    if len(values) > 1
                    else 0.0
                ),
                "bootstrap_95_ci": paired_bootstrap_mean_ci(
                    values,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                ),
            }
            for name, values in deltas.items()
        },
    }
    improvement = {}
    for name, values in deltas.items():
        delta_array = np.asarray(values, dtype=np.float64)
        improved = int(np.count_nonzero(delta_array > 0.0))
        tied = int(np.count_nonzero(delta_array == 0.0))
        regressed = int(delta_array.size - improved - tied)
        improvement[name] = {
            "improved_pairs": improved,
            "tied_pairs": tied,
            "regressed_pairs": regressed,
            "percentage_improved": 100.0 * improved / delta_array.size,
        }

    secret_psnr_deltas = np.asarray(deltas["secret_psnr_db"], dtype=np.float64)
    return {
        "aggregate_metrics": aggregate,
        "improvement_by_metric": improvement,
        "secret_psnr_delta_db": {
            "mean": float(secret_psnr_deltas.mean()),
            "percentage_improved": improvement["secret_psnr_db"]["percentage_improved"],
            "bootstrap_95_ci": paired_bootstrap_mean_ci(
                secret_psnr_deltas,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
        },
    }


def _evaluate_one(model: nn.Module, cover: Tensor, secret: Tensor) -> dict[str, float]:
    outputs = model(cover, secret)
    stego = outputs["stego"].float()
    revealed = outputs["revealed_secret"].float()
    tensors = {
        **image_metrics(cover.float(), stego, "cover"),
        **image_metrics(secret.float(), revealed, "secret"),
    }
    values = {name: float(tensors[name].item()) for name in METRIC_NAMES}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("checkpoint produced a non-finite per-image metric")
    return values


@torch.inference_mode()
def compare_models(
    baseline_model: nn.Module,
    candidate_model: nn.Module,
    pairs: list[PairSpec],
    *,
    image_size: int,
    device: torch.device,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Evaluate two models on identical pairs, one pair at a time."""

    if not pairs:
        raise ValueError("at least one pair is required")
    baseline_model.float().eval()
    candidate_model.float().eval()
    records: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        cover = load_rgb_image(pair.cover_path, image_size).unsqueeze(0).to(device)
        secret = load_rgb_image(pair.secret_path, image_size).unsqueeze(0).to(device)
        baseline_metrics = _evaluate_one(baseline_model, cover, secret)
        candidate_metrics = _evaluate_one(candidate_model, cover, secret)
        records.append(
            {
                "index": index,
                "id": pair.identifier,
                "cover": pair.cover_reference,
                "secret": pair.secret_reference,
                "cover_sha256": pair.cover_sha256 or _sha256(pair.cover_path),
                "secret_sha256": pair.secret_sha256 or _sha256(pair.secret_path),
                "baseline_metrics": baseline_metrics,
                "candidate_metrics": candidate_metrics,
                "delta_candidate_minus_baseline": {
                    name: candidate_metrics[name] - baseline_metrics[name] for name in METRIC_NAMES
                },
            }
        )
    return {
        **summarize_records(
            records,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "pairs": records,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Prefer a repository-relative provenance path when one is available."""

    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _validate_checkpoint_configs(baseline_config: Any, candidate_config: Any) -> int:
    for label, config in (("baseline", baseline_config), ("candidate", candidate_config)):
        if config.model.payload_bits is not None:
            raise ValueError(f"{label} checkpoint uses bit payloads, not RGB secret images")
        if config.model.secret_size != config.data.image_size:
            raise ValueError(f"{label} checkpoint has mismatched cover and secret sizes")
    if baseline_config.data.image_size != candidate_config.data.image_size:
        raise ValueError("baseline and candidate checkpoints require different image sizes")
    return int(baseline_config.data.image_size)


def compare_checkpoints(
    baseline_path: str | Path,
    candidate_path: str | Path,
    manifest_path: str | Path,
    *,
    device: torch.device,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Load checkpoints and produce a traceable paired-comparison artifact."""

    baseline_path = Path(baseline_path).expanduser().resolve()
    candidate_path = Path(candidate_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    pairs = load_manifest(manifest_path)
    baseline_model, baseline_config = load_model_from_checkpoint(baseline_path, device)
    candidate_model, candidate_config = load_model_from_checkpoint(candidate_path, device)
    image_size = _validate_checkpoint_configs(baseline_config, candidate_config)
    comparison = compare_models(
        baseline_model,
        candidate_model,
        pairs,
        image_size=image_size,
        device=device,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "schema_version": 1,
        "comparison_direction": "candidate minus baseline; positive is better",
        "manifest": {
            "path": _display_path(manifest_path),
            "file_sha256": _sha256(manifest_path),
            "declared_sha256": json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "manifest_sha256"
            ),
        },
        "pair_count": len(pairs),
        "image_size_px": [image_size, image_size],
        "checkpoints": {
            "baseline": {
                "path": _display_path(baseline_path),
                "sha256": _sha256(baseline_path),
            },
            "candidate": {
                "path": _display_path(candidate_path),
                "sha256": _sha256(candidate_path),
            },
        },
        "bootstrap": {
            "unit": "manifest cover/secret pair",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "evaluation_protocol": {
            "precision": "float32 (autocast disabled, TF32 disabled)",
            "batch_size": 1,
            "resize": "EXIF transpose, RGB conversion, bicubic center fit",
            "input_range": [-1.0, 1.0],
            "jpeg_attack": None,
        },
        **comparison,
    }


def write_json_atomic(payload: dict[str, Any], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="baseline checkpoint")
    parser.add_argument("--candidate", required=True, help="candidate checkpoint")
    parser.add_argument("--manifest", required=True, help="explicit cover/secret JSON manifest")
    parser.add_argument("--output", default="results/checkpoint_comparison.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = select_device(args.device)
    payload = compare_checkpoints(
        args.baseline,
        args.candidate,
        args.manifest,
        device=device,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    destination = write_json_atomic(payload, args.output)
    print(
        json.dumps(
            {
                "output": destination.resolve().as_posix(),
                "pair_count": payload["pair_count"],
                "secret_psnr_delta_db": payload["secret_psnr_delta_db"],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
