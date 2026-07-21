"""Reconcile committed results and independently check a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from stego.cli import _datasets
from stego.metrics import psnr
from stego.training import load_model_from_checkpoint, select_device

RECONCILIATION_FIELDS = (
    ("cover_psnr_db", "cover_psnr_db"),
    ("cover_ssim", "cover_ssim"),
    ("secret_psnr_db", "secret_clean_psnr_db"),
    ("secret_ssim", "secret_clean_ssim"),
    ("jpeg_q50_psnr_db", "secret_q50_psnr_db"),
    ("jpeg_q50_ssim", "secret_q50_ssim"),
)
ADAPTER_KEY_FRAGMENTS = (".attention.", ".band_gate.", ".router.")
ZERO_INITIALIZED_NEW_ADAPTER_FRAGMENTS = (".router.skew_predictor.2.",)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _comparison_models(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    raw_models = comparison.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("comparison must contain a non-empty models list")
    models: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, model in enumerate(raw_models):
        if not isinstance(model, dict):
            raise ValueError(f"comparison model {index} must be an object")
        name = model.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"comparison model {index} needs a non-empty name")
        if name in names:
            raise ValueError(f"comparison model name is duplicated: {name}")
        names.add(name)
        models.append(model)
    return models


def _named_model(
    models: list[dict[str, Any]],
    name: str,
    role: str,
) -> dict[str, Any]:
    matches = [model for model in models if model["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"{role} model {name!r} must match exactly one comparison row")
    return matches[0]


def _metadata_name(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _model_reconciliation(
    metrics: dict[str, Any],
    model: dict[str, Any],
) -> tuple[bool, list[str]]:
    compared: list[str] = []
    reconciled = True
    for metric_name, model_field in RECONCILIATION_FIELDS:
        if metric_name not in metrics or model_field not in model:
            continue
        try:
            benchmark_value = float(metrics[metric_name])
            comparison_value = float(model[model_field])
        except (TypeError, ValueError):
            return False, compared
        compared.append(f"{metric_name}↔{model_field}")
        reconciled = reconciled and math.isclose(
            benchmark_value,
            comparison_value,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    required = {"cover_psnr_db", "secret_psnr_db"}
    compared_metrics = {item.split("↔", maxsplit=1)[0] for item in compared}
    return reconciled and required.issubset(compared_metrics), compared


def resolve_selected_model(
    benchmark: dict[str, Any],
    comparison: dict[str, Any],
    explicit_name: str | None = None,
) -> dict[str, Any]:
    """Select a comparison row by name/marker or a unique metric reconciliation."""

    models = _comparison_models(comparison)
    benchmark_model = benchmark.get("model")
    benchmark_model_name = (
        benchmark_model.get("name") if isinstance(benchmark_model, dict) else None
    )
    selected_name = explicit_name or _metadata_name(
        comparison,
        ("selected_model_name", "selected_model", "candidate_model_name"),
    )
    if selected_name is None:
        selected_name = _metadata_name(benchmark, ("selected_model_name",))
    if selected_name is None and isinstance(benchmark_model_name, str):
        selected_name = benchmark_model_name
    if selected_name is not None:
        return _named_model(models, selected_name, "selected")

    marked = [model for model in models if model.get("selected") is True]
    if len(marked) == 1:
        return marked[0]
    if len(marked) > 1:
        raise ValueError("comparison marks more than one model as selected")

    metrics = benchmark.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("benchmark metrics must be an object")
    matches = [model for model in models if _model_reconciliation(metrics, model)[0]]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        "selected model is ambiguous; set selected_model_name or pass --selected-model"
    )


def resolve_upstream_model(
    comparison: dict[str, Any],
    explicit_name: str | None = None,
) -> dict[str, Any] | None:
    """Resolve an upstream row only when the evidence explicitly identifies one."""

    models = _comparison_models(comparison)
    upstream_name = explicit_name or _metadata_name(
        comparison,
        ("upstream_model_name", "reference_model_name", "baseline_model_name"),
    )
    if upstream_name is not None:
        return _named_model(models, upstream_name, "upstream")
    marked = [
        model
        for model in models
        if str(model.get("role", "")).lower() in {"upstream", "reference", "baseline"}
    ]
    if len(marked) == 1:
        return marked[0]
    if len(marked) > 1:
        raise ValueError("comparison marks more than one model as upstream/reference")
    return None


def model_pareto_deltas(
    candidate: dict[str, Any],
    upstream: dict[str, Any],
) -> dict[str, float]:
    """Return candidate-minus-upstream PSNR deltas for the clean Pareto objectives."""

    return {
        "cover_psnr_db": float(candidate["cover_psnr_db"])
        - float(upstream["cover_psnr_db"]),
        "secret_psnr_db": float(candidate["secret_clean_psnr_db"])
        - float(upstream["secret_clean_psnr_db"]),
    }


def evaluation_precision(payload: dict[str, Any]) -> str | None:
    """Extract an explicit evaluation-precision declaration from supported schemas."""

    direct = payload.get("evaluation_precision", payload.get("precision"))
    if isinstance(direct, str) and direct.strip():
        return direct
    for section_name in ("evaluation_protocol", "evaluation", "protocol"):
        section = payload.get(section_name)
        if isinstance(section, dict):
            value = section.get("precision")
            if isinstance(value, str) and value.strip():
                return value
    return None


def is_explicit_fp32(precision: str | None) -> bool:
    """Accept only an explicit full-float declaration, never mixed/reduced precision."""

    if precision is None:
        return False
    normalized = precision.lower().replace(" ", "")
    reduced_markers = ("bfloat16", "bf16", "float16", "fp16", "mixed", "autocastenabled")
    return (
        ("float32" in normalized or "fp32" in normalized)
        and not any(marker in normalized for marker in reduced_markers)
    )


def paired_pareto_deltas(paired: dict[str, Any]) -> dict[str, float]:
    """Read the clean candidate-minus-baseline paired mean deltas."""

    direction = str(paired.get("comparison_direction", "")).lower()
    if "candidate minus baseline" not in direction or "positive is better" not in direction:
        raise ValueError("paired comparison direction must be candidate minus baseline")
    try:
        aggregate = paired["aggregate_metrics"]["candidate_minus_baseline"]
        deltas = {
            "cover_psnr_db": float(aggregate["cover_psnr_db"]["mean"]),
            "secret_psnr_db": float(aggregate["secret_psnr_db"]["mean"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("paired comparison lacks clean aggregate Pareto deltas") from error
    if not all(math.isfinite(value) for value in deltas.values()):
        raise ValueError("paired Pareto deltas must be finite")
    return deltas


def paired_evidence_integrity(paired: dict[str, Any]) -> tuple[bool, str]:
    """Reconcile pair count and the secret-PSNR summary with pair-level evidence."""

    raw_pairs = paired.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        return False, "no pair-level records"
    try:
        deltas = [
            float(record["delta_candidate_minus_baseline"]["secret_psnr_db"])
            for record in raw_pairs
        ]
        declared_count = int(paired["pair_count"])
        summary = paired["secret_psnr_delta_db"]
        reported_mean = float(summary["mean"])
        interval = summary["bootstrap_95_ci"]
        lower = float(interval["lower"])
        upper = float(interval["upper"])
    except (KeyError, TypeError, ValueError):
        return False, "missing pair-count, delta, or bootstrap fields"
    observed_mean = sum(deltas) / len(deltas)
    valid = (
        declared_count == len(deltas)
        and all(math.isfinite(value) for value in deltas)
        and math.isclose(observed_mean, reported_mean, rel_tol=0.0, abs_tol=1e-9)
        and lower <= reported_mean <= upper
    )
    return (
        valid,
        f"n={len(deltas)}, mean={reported_mean:+.4f} dB, "
        f"95% CI=[{lower:+.4f}, {upper:+.4f}]",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint {path} does not contain a model state")
    return state


def changed_adapter_parameter_count(
    warm_state: dict[str, torch.Tensor],
    selected_state: dict[str, torch.Tensor],
) -> int:
    """Count changed shared adapters plus learned zero-initialized router outputs."""

    changes: list[torch.Tensor] = []
    for key, selected_value in selected_state.items():
        if not isinstance(selected_value, torch.Tensor):
            continue
        warm_value = warm_state.get(key)
        if (
            any(fragment in key for fragment in ADAPTER_KEY_FRAGMENTS)
            and isinstance(warm_value, torch.Tensor)
            and selected_value.shape == warm_value.shape
        ):
            changes.append((selected_value.float() - warm_value.float()).flatten())
        elif (
            warm_value is None
            and any(
                fragment in key for fragment in ZERO_INITIALIZED_NEW_ADAPTER_FRAGMENTS
            )
        ):
            changes.append(selected_value.float().flatten())
    return int((torch.cat(changes) != 0).sum()) if changes else 0


def _paired_checkpoint_identity(
    paired: dict[str, Any],
    warmstart: Path,
    checkpoint: Path,
) -> tuple[bool, str]:
    try:
        baseline_sha = str(paired["checkpoints"]["baseline"]["sha256"])
        candidate_sha = str(paired["checkpoints"]["candidate"]["sha256"])
    except (KeyError, TypeError):
        return False, "paired evidence lacks checkpoint hashes"
    actual_baseline = _sha256(warmstart)
    actual_candidate = _sha256(checkpoint)
    valid = baseline_sha == actual_baseline and candidate_sha == actual_candidate
    return valid, f"baseline={baseline_sha[:12]}…, candidate={candidate_sha[:12]}…"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("results/benchmark_256.json"))
    parser.add_argument("--comparison", type=Path, default=Path("results/comparison_256.json"))
    parser.add_argument("--data-profile", type=Path, default=Path("results/data_profile.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--selected-model")
    parser.add_argument("--upstream-model")
    parser.add_argument(
        "--paired-comparison",
        type=Path,
        help=(
            "optional paired candidate-vs-upstream artifact "
            "(auto-detects results/paired_final.json)"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/validation_report.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    benchmark = _load(args.benchmark)
    comparison = _load(args.comparison)
    profile = _load(args.data_profile)
    metrics = benchmark.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("benchmark metrics must be an object")
    selected_model = resolve_selected_model(
        benchmark,
        comparison,
        explicit_name=args.selected_model,
    )
    selected_name = str(selected_model["name"])
    reconciled, reconciled_fields = _model_reconciliation(metrics, selected_model)
    upstream_model = resolve_upstream_model(comparison, explicit_name=args.upstream_model)

    default_paired = Path("results/paired_final.json")
    paired_path = args.paired_comparison
    if paired_path is None and default_paired.is_file():
        paired_path = default_paired
    if args.paired_comparison is not None and not args.paired_comparison.is_file():
        raise FileNotFoundError(f"paired comparison does not exist: {args.paired_comparison}")
    paired = _load(paired_path) if paired_path is not None else None

    jpeg_psnr: list[float] = []
    for quality in (50, 60, 70, 80, 90):
        value = metrics.get(f"jpeg_q{quality}_psnr_db")
        if value is not None:
            jpeg_psnr.append(float(value))
    jpeg_failure_exposed = len(jpeg_psnr) == 5 and max(jpeg_psnr) < 12.0
    data_checks = profile.get("checks")
    if not isinstance(data_checks, dict):
        raise ValueError("data profile checks must be an object")
    data_valid = bool(data_checks) and all(bool(value) for value in data_checks.values())

    nonfinite_steps = 0.0
    with args.history.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            nonfinite_steps = max(nonfinite_steps, float(row["nonfinite_steps"]))

    device = select_device(args.device)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    model = model.float().eval()
    data_roots = benchmark.get("data_roots")
    if not isinstance(data_roots, list) or not all(
        isinstance(path, str) for path in data_roots
    ):
        raise ValueError("benchmark data_roots must be a list of paths")
    _, validation, _, discovered = _datasets(data_roots, config)
    if len(validation) < 4:
        raise ValueError("at least four validation pairs are required for checkpoint checks")
    first_pairs = [validation[index] for index in range(4)]
    cover = torch.stack([pair[0] for pair in first_pairs]).to(
        device=device, dtype=torch.float32
    )
    secret = torch.stack([pair[1] for pair in first_pairs]).to(
        device=device, dtype=torch.float32
    )
    with torch.inference_mode():
        outputs = model(cover, secret)
    stego = outputs["stego"]
    revealed = outputs["revealed_secret"]
    mse = (cover - stego.float()).square().flatten(start_dim=1).mean(dim=1)
    manual_psnr = 10.0 * torch.log10(4.0 / mse)
    metric_psnr = psnr(cover, stego.float())
    psnr_difference = (manual_psnr - metric_psnr).abs().max().item()
    checkpoint_shapes_valid = (
        tuple(cover.shape[-2:]) == (256, 256)
        and tuple(secret.shape[-2:]) == (256, 256)
        and tuple(stego.shape) == tuple(cover.shape)
        and tuple(revealed.shape) == tuple(secret.shape)
    )
    checkpoint_forward_fp32 = stego.dtype == torch.float32 and revealed.dtype == torch.float32

    warm_state = _checkpoint_state(args.warmstart)
    selected_state = _checkpoint_state(args.checkpoint)
    changed_adapter_parameters = changed_adapter_parameter_count(
        warm_state,
        selected_state,
    )

    precision = evaluation_precision(benchmark)
    clean_cover = float(metrics["cover_psnr_db"])
    clean_secret = float(metrics["secret_psnr_db"])
    clean_fp32_target = (
        is_explicit_fp32(precision) and clean_cover >= 38.0 and clean_secret >= 35.0
    )
    checks: list[tuple[str, bool, str]] = [
        (
            "Selected-model reconciliation",
            reconciled,
            f"{selected_name}; {len(reconciled_fields)} named fields exact within 1e-9",
        ),
        ("Dataset audit", data_valid, "Readable, unique, disjoint split"),
        ("Checkpoint tensor shapes", checkpoint_shapes_valid, "256×256 cover and secret"),
        (
            "Checkpoint FP32 forward",
            checkpoint_forward_fp32,
            f"stego={stego.dtype}, revealed={revealed.dtype}",
        ),
        (
            "Independent PSNR calculation",
            psnr_difference < 1e-5,
            f"max Δ={psnr_difference:.2e} dB",
        ),
        (
            "Corrected clean FP32 target",
            clean_fp32_target,
            f"precision={precision!r}; cover={clean_cover:.3f}, secret={clean_secret:.3f} dB",
        ),
        (
            "Local adapters updated",
            changed_adapter_parameters > 0,
            f"changed shared adapter parameters={changed_adapter_parameters:,}",
        ),
        (
            "JPEG limitation exposed",
            jpeg_failure_exposed,
            "all five real-JPEG PSNR values are present and <12 dB",
        ),
        (
            "Numerical stability",
            nonfinite_steps == 0,
            f"non-finite steps={nonfinite_steps:.0f}",
        ),
    ]

    model_deltas: dict[str, float] | None = None
    if upstream_model is not None:
        model_deltas = model_pareto_deltas(selected_model, upstream_model)
        checks.append(
            (
                "Aggregate candidate-vs-upstream Pareto",
                all(delta > 0.0 for delta in model_deltas.values()),
                f"cover={model_deltas['cover_psnr_db']:+.4f} dB, "
                f"secret={model_deltas['secret_psnr_db']:+.4f} dB",
            )
        )

    paired_deltas: dict[str, float] | None = None
    if paired is not None:
        paired_deltas = paired_pareto_deltas(paired)
        paired_integrity, paired_integrity_evidence = paired_evidence_integrity(paired)
        paired_identity, paired_identity_evidence = _paired_checkpoint_identity(
            paired,
            args.warmstart,
            args.checkpoint,
        )
        paired_precision = evaluation_precision(paired)
        checks.extend(
            [
                ("Paired evidence integrity", paired_integrity, paired_integrity_evidence),
                (
                    "Paired evaluation protocol",
                    is_explicit_fp32(paired_precision),
                    f"precision={paired_precision!r}",
                ),
                (
                    "Paired candidate-vs-upstream Pareto",
                    all(delta > 0.0 for delta in paired_deltas.values()),
                    f"cover={paired_deltas['cover_psnr_db']:+.4f} dB, "
                    f"secret={paired_deltas['secret_psnr_db']:+.4f} dB",
                ),
                ("Paired checkpoint identity", paired_identity, paired_identity_evidence),
            ]
        )

    passed = all(check[1] for check in checks)
    rows = "\n".join(
        f"| {name} | {'PASS' if result else 'FAIL'} | {evidence} |"
        for name, result, evidence in checks
    )
    evaluated_images = int(float(metrics.get("evaluated_images", 0)))
    benchmark_model = benchmark.get("model")
    benchmark_model = benchmark_model if isinstance(benchmark_model, dict) else {}
    local_steps = benchmark_model.get("local_gate_steps", benchmark_model.get("local_steps"))
    step_text = f", {local_steps} local adapter steps" if local_steps is not None else ""
    scope = f"{selected_name}{step_text}; {evaluated_images}-pair aggregate benchmark"

    pareto_rows = ""
    if model_deltas is not None:
        pareto_rows += (
            f"| Aggregate cover PSNR delta vs upstream | "
            f"{model_deltas['cover_psnr_db']:+.4f} dB |\n"
            f"| Aggregate secret PSNR delta vs upstream | "
            f"{model_deltas['secret_psnr_db']:+.4f} dB |\n"
        )
    if paired_deltas is not None:
        pareto_rows += (
            f"| Paired cover PSNR mean delta vs upstream | "
            f"{paired_deltas['cover_psnr_db']:+.4f} dB |\n"
            f"| Paired secret PSNR mean delta vs upstream | "
            f"{paired_deltas['secret_psnr_db']:+.4f} dB |\n"
        )

    validation_counts = benchmark.get("validation_source_counts")
    source_caveat = ""
    if isinstance(validation_counts, dict) and validation_counts:
        smallest_source, smallest_count = min(
            validation_counts.items(), key=lambda item: int(item[1])
        )
        source_total = sum(int(value) for value in validation_counts.values())
        source_caveat = (
            f"- Only {int(smallest_count)}/{source_total} benchmark images are from "
            f"{smallest_source}; source-specific claims are unsupported.\n"
        )
    if paired is None:
        uncertainty_caveat = (
            "- No paired final-comparison artifact was supplied; confidence intervals "
            "and pair-level win rates are not validated.\n"
        )
    else:
        pair_count = int(paired["pair_count"])
        uncertainty_caveat = (
            f"- The paired bootstrap covers one fixed {pair_count}-pair cohort; "
            "it does not measure training-seed variability.\n"
        )
    local_parameters = benchmark_model.get("locally_trained_gate_parameters")
    adaptation_caveat = (
        f"- Local adaptation updates {int(local_parameters):,} declared adapter parameters.\n"
        if local_parameters is not None
        else "- The checkpoint is a local adaptation of the disclosed warm start.\n"
    )
    q50_psnr = float(metrics["jpeg_q50_psnr_db"])
    report = f"""# Result validation report

**Decision:** {'Share with caveats' if passed else 'Do not share'}

**Scope:** {scope}

**Data discovered:** {discovered:,} images across the declared benchmark roots

## Checks

| Check | Result | Evidence |
|---|---:|---|
{rows}

## Reconciled headline metrics

| Metric | Value |
|---|---:|
| Evaluation precision | {precision or 'not declared'} |
| Cover → stego PSNR | {clean_cover:.3f} dB |
| Cover → stego SSIM | {float(metrics['cover_ssim']):.4f} |
| Secret recovery PSNR (clean) | {clean_secret:.3f} dB |
| Secret recovery SSIM (clean) | {float(metrics['secret_ssim']):.4f} |
| Secret recovery PSNR (real JPEG QF-50) | {q50_psnr:.3f} dB |
| Secret recovery SSIM (real JPEG QF-50) | {float(metrics['jpeg_q50_ssim']):.4f} |
{pareto_rows}
## Caveats

{uncertainty_caveat}{source_caveat}- Real JPEG destroys recovery (about {q50_psnr:.1f} dB
  at QF-50); this checkpoint is clean-channel only.
- The invertible backbone uses disclosed public HiNet weights; this is not training from scratch.
{adaptation_caveat}- Similarity percentage is MAE-derived fidelity, not exact-pixel extraction
  accuracy.
- No classical steganography baseline was reimplemented on the paired cohort; broad
  superiority claims are out of scope.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
