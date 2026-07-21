"""Reconcile committed results and independently check a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from stego.cli import _datasets
from stego.metrics import psnr
from stego.training import load_model_from_checkpoint, select_device


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("results/benchmark_256.json"))
    parser.add_argument("--comparison", type=Path, default=Path("results/comparison_256.json"))
    parser.add_argument("--data-profile", type=Path, default=Path("results/data_profile.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmstart", required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/validation_report.md"))
    args = parser.parse_args()

    benchmark = _load(args.benchmark)
    comparison = _load(args.comparison)
    profile = _load(args.data_profile)
    metrics = benchmark["metrics"]
    models = comparison["models"]
    assert isinstance(metrics, dict) and isinstance(models, list)
    final_model = models[-1]

    reconciled = all(
        math.isclose(float(metrics[metric]), float(final_model[field]), abs_tol=1e-9)
        for metric, field in (
            ("cover_psnr_db", "cover_psnr_db"),
            ("secret_psnr_db", "secret_clean_psnr_db"),
            ("jpeg_q50_psnr_db", "secret_q50_psnr_db"),
        )
    )
    jpeg_psnr = [float(metrics[f"jpeg_q{quality}_psnr_db"]) for quality in (50, 60, 70, 80, 90)]
    jpeg_failure_exposed = max(jpeg_psnr) < 12.0
    data_checks = profile["checks"]
    assert isinstance(data_checks, dict)
    data_valid = all(bool(value) for value in data_checks.values())

    nonfinite_steps = 0.0
    with args.history.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            nonfinite_steps = max(nonfinite_steps, float(row["nonfinite_steps"]))

    device = select_device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    _, validation, _, discovered = _datasets(
        [str(path) for path in benchmark["data_roots"]], config
    )
    first_pairs = [validation[index] for index in range(4)]
    cover = torch.stack([pair[0] for pair in first_pairs]).to(device)
    secret = torch.stack([pair[1] for pair in first_pairs]).to(device)
    with torch.inference_mode():
        outputs = model(cover, secret)
    mse = (cover - outputs["stego"]).square().flatten(start_dim=1).mean(dim=1)
    manual_psnr = 10.0 * torch.log10(4.0 / mse)
    metric_psnr = psnr(cover, outputs["stego"])
    psnr_difference = (manual_psnr - metric_psnr).abs().max().item()
    checkpoint_shapes_valid = (
        tuple(cover.shape[-2:]) == (256, 256)
        and tuple(secret.shape[-2:]) == (256, 256)
        and tuple(outputs["stego"].shape) == tuple(cover.shape)
    )
    warm_state = torch.load(args.warmstart, map_location="cpu", weights_only=True)["model"]
    selected_state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    gate_keys = [
        key for key in selected_state if ".attention." in key or ".band_gate." in key
    ]
    gate_changes = torch.cat(
        [(selected_state[key].float() - warm_state[key].float()).flatten() for key in gate_keys]
    )
    changed_gate_parameters = int((gate_changes != 0).sum())

    checks = [
        ("Benchmark/comparison reconciliation", reconciled, "Exact within 1e-9 dB"),
        ("Dataset audit", data_valid, "Readable, unique, disjoint split"),
        ("Checkpoint tensor shapes", checkpoint_shapes_valid, "256×256 cover and secret"),
        ("Independent PSNR calculation", psnr_difference < 1e-5, f"max Δ={psnr_difference:.2e} dB"),
        ("Aggregate cover target", float(metrics["cover_psnr_db"]) >= 38.0, ">=38 dB"),
        ("Aggregate secret target", float(metrics["secret_psnr_db"]) >= 35.0, ">=35 dB"),
        (
            "Novel gates updated",
            changed_gate_parameters > 0,
            f"changed parameters={changed_gate_parameters:,}",
        ),
        ("JPEG limitation exposed", jpeg_failure_exposed, "all real-JPEG PSNR values <12 dB"),
        ("Numerical stability", nonfinite_steps == 0, f"non-finite steps={nonfinite_steps:.0f}"),
    ]
    passed = all(check[1] for check in checks)
    rows = "\n".join(
        f"| {name} | {'PASS' if result else 'FAIL'} | {evidence} |"
        for name, result, evidence in checks
    )
    report = f"""# Result validation report

**Decision:** {'Share with caveats' if passed else 'Do not share'}

**Scope:** HiNet-warm-started, 100-step gates-only checkpoint; 200-pair benchmark

**Data discovered:** {discovered:,} images across COCO val2017 and DIV2K validation

## Checks

| Check | Result | Evidence |
|---|---:|---|
{rows}

## Reconciled headline metrics

| Metric | Value |
|---|---:|
| Cover → stego PSNR | {float(metrics['cover_psnr_db']):.3f} dB |
| Cover → stego SSIM | {float(metrics['cover_ssim']):.4f} |
| Secret recovery PSNR (clean) | {float(metrics['secret_psnr_db']):.3f} dB |
| Secret recovery SSIM (clean) | {float(metrics['secret_ssim']):.4f} |
| Secret recovery PSNR (real JPEG QF-50) | {float(metrics['jpeg_q50_psnr_db']):.3f} dB |
| Secret recovery SSIM (real JPEG QF-50) | {float(metrics['jpeg_q50_ssim']):.4f} |

## Caveats

- Results use one seeded 90/10 split; there are no multi-seed confidence intervals.
- Only 3/200 validation images are from DIV2K; source-specific claims are unsupported.
- Real JPEG destroys recovery (about 11.2 dB); this checkpoint is clean-channel only.
- The invertible backbone uses disclosed public HiNet weights; this is not training from scratch.
- Local adaptation updates 123,024 attention/band-gate parameters for 100 steps.
- Similarity percentage is MAE-derived fidelity, not exact-pixel extraction accuracy.
- No external baseline was reimplemented, so superiority claims are out of scope.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
