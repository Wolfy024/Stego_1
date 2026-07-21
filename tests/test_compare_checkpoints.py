from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "compare_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("compare_checkpoints", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not import {SCRIPT_PATH}")
COMPARE_CHECKPOINTS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPARE_CHECKPOINTS
SPEC.loader.exec_module(COMPARE_CHECKPOINTS)

METRIC_NAMES = COMPARE_CHECKPOINTS.METRIC_NAMES
compare_models = COMPARE_CHECKPOINTS.compare_models
load_manifest = COMPARE_CHECKPOINTS.load_manifest
paired_bootstrap_mean_ci = COMPARE_CHECKPOINTS.paired_bootstrap_mean_ci
summarize_records = COMPARE_CHECKPOINTS.summarize_records
write_json_atomic = COMPARE_CHECKPOINTS.write_json_atomic


class PerturbedIdentityModel(nn.Module):
    """Deterministic test double whose lower amplitude has better fidelity."""

    def __init__(self, amplitude: float) -> None:
        super().__init__()
        self.amplitude = amplitude

    def forward(self, cover: Tensor, secret: Tensor) -> dict[str, Tensor]:
        height, width = cover.shape[-2:]
        rows = torch.arange(height, device=cover.device).view(1, 1, height, 1)
        columns = torch.arange(width, device=cover.device).view(1, 1, 1, width)
        checkerboard = ((rows + columns) % 2).mul(2).sub(1).to(cover.dtype)
        return {
            "stego": (cover + self.amplitude * checkerboard).clamp(-1.0, 1.0),
            "revealed_secret": (secret - self.amplitude * checkerboard).clamp(-1.0, 1.0),
        }


def _write_image(path: Path, offset: int) -> None:
    rows, columns = np.indices((40, 48))
    red = (80 + offset + rows) % 180
    green = (90 + offset + columns) % 180
    blue = (100 + offset + rows + columns) % 180
    array = np.stack((red, green, blue), axis=-1).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


class CompareCheckpointTests(unittest.TestCase):
    def test_loads_content_addressed_rooted_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            manifests = root / "results" / "manifests"
            images.mkdir()
            manifests.mkdir(parents=True)
            _write_image(images / "cover.png", 0)
            _write_image(images / "secret.png", 12)
            cover_sha = COMPARE_CHECKPOINTS._sha256(images / "cover.png")
            secret_sha = COMPARE_CHECKPOINTS._sha256(images / "secret.png")
            payload = {
                "schema_version": 1,
                "data_roots": {"images": "../../images"},
                "pairs": [
                    {
                        "pair_id": "fresh-0001",
                        "cover_path": "images/cover.png",
                        "cover_sha256": cover_sha,
                        "secret_path": "images/secret.png",
                        "secret_sha256": secret_sha,
                    }
                ],
            }
            payload["manifest_sha256"] = COMPARE_CHECKPOINTS._canonical_sha256(payload)
            manifest = manifests / "dev.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            pairs = load_manifest(manifest)

            self.assertEqual(pairs[0].identifier, "fresh-0001")
            self.assertEqual(pairs[0].cover_path, (images / "cover.png").resolve())
            self.assertEqual(pairs[0].cover_sha256, cover_sha)

    def test_rejects_tampered_content_addressed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_image(root / "cover.png", 0)
            _write_image(root / "secret.png", 12)
            payload = {
                "pairs": [
                    {
                        "id": "pair-1",
                        "cover": "cover.png",
                        "secret": "secret.png",
                    }
                ],
                "manifest_sha256": "0" * 64,
            }
            manifest = root / "pairs.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "manifest_sha256 mismatch"):
                load_manifest(manifest)

    def test_manifest_resolves_relative_paths_and_preserves_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cover.png").write_bytes(b"cover")
            (root / "secret.png").write_bytes(b"secret")
            manifest = root / "pairs.json"
            manifest.write_text(
                json.dumps(
                    {
                        "pairs": [
                            {
                                "id": "held-out-7",
                                "cover": "cover.png",
                                "secret": "secret.png",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            pairs = load_manifest(manifest)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].identifier, "held-out-7")
            self.assertEqual(pairs[0].cover_path, (root / "cover.png").resolve())
            self.assertEqual(pairs[0].cover_reference, "cover.png")

    def test_manifest_rejects_duplicate_statistical_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cover.png").write_bytes(b"cover")
            (root / "secret.png").write_bytes(b"secret")
            manifest = root / "pairs.json"
            manifest.write_text(
                json.dumps(
                    {
                        "pairs": [
                            {"id": "a", "cover": "cover.png", "secret": "secret.png"},
                            {"id": "b", "cover": "cover.png", "secret": "secret.png"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate cover/secret pair"):
                load_manifest(manifest)

    def test_paired_bootstrap_is_seeded_and_contains_observed_mean(self) -> None:
        deltas = np.array([0.5, 1.0, 1.5, 3.0], dtype=np.float64)

        first = paired_bootstrap_mean_ci(deltas, samples=1_000, seed=17)
        second = paired_bootstrap_mean_ci(deltas, samples=1_000, seed=17)

        self.assertEqual(first, second)
        self.assertLessEqual(first["lower"], deltas.mean())
        self.assertGreaterEqual(first["upper"], deltas.mean())
        self.assertEqual(first["method"], "paired percentile bootstrap")

    def test_summary_reports_paired_delta_and_percentage_improved(self) -> None:
        secret_deltas = (-1.0, 0.0, 2.0)
        records = []
        for index, secret_delta in enumerate(secret_deltas):
            baseline = {name: 10.0 + index for name in METRIC_NAMES}
            candidate = {name: value + 0.25 for name, value in baseline.items()}
            candidate["secret_psnr_db"] = baseline["secret_psnr_db"] + secret_delta
            records.append(
                {
                    "baseline_metrics": baseline,
                    "candidate_metrics": candidate,
                }
            )

        summary = summarize_records(records, bootstrap_samples=500, bootstrap_seed=9)

        improvement = summary["improvement_by_metric"]["secret_psnr_db"]
        self.assertEqual(improvement["improved_pairs"], 1)
        self.assertEqual(improvement["tied_pairs"], 1)
        self.assertEqual(improvement["regressed_pairs"], 1)
        self.assertAlmostEqual(improvement["percentage_improved"], 100.0 / 3.0)
        self.assertAlmostEqual(
            summary["aggregate_metrics"]["candidate_minus_baseline"]["secret_psnr_db"]["mean"],
            sum(secret_deltas) / len(secret_deltas),
        )

    def test_compare_models_keeps_per_pair_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_image(root / "cover.png", 0)
            _write_image(root / "secret.png", 12)
            manifest = root / "pairs.json"
            manifest.write_text(
                json.dumps(
                    {
                        "pairs": [
                            {
                                "id": "pair-1",
                                "cover": "cover.png",
                                "secret": "secret.png",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            pairs = load_manifest(manifest)

            comparison = compare_models(
                PerturbedIdentityModel(0.08),
                PerturbedIdentityModel(0.02),
                pairs,
                image_size=32,
                device=torch.device("cpu"),
                bootstrap_samples=200,
                bootstrap_seed=23,
            )

            record = comparison["pairs"][0]
            self.assertEqual(record["id"], "pair-1")
            self.assertGreater(
                record["delta_candidate_minus_baseline"]["secret_psnr_db"],
                0.0,
            )
            self.assertEqual(
                comparison["secret_psnr_delta_db"]["percentage_improved"],
                100.0,
            )
            interval = comparison["secret_psnr_delta_db"]["bootstrap_95_ci"]
            self.assertAlmostEqual(interval["lower"], interval["upper"])

    def test_atomic_json_writer_emits_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "comparison.json"
            payload = {"pair_count": 2, "delta": 0.75}

            written = write_json_atomic(payload, destination)

            self.assertEqual(written, destination)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), payload)
            self.assertFalse(destination.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
