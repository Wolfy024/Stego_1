"""Create README-ready charts from committed benchmark evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2563EB"
ORANGE = "#EA580C"
GREEN = "#15803D"
INK = "#172033"
MUTED = "#526173"
GRID = "#D7DEE8"
MODEL_COLORS = (ORANGE, BLUE, GREEN)

COMPARISON_CATEGORIES = ("Cover → stego", "Secret (clean)", "Secret (QF-50)")
COMPARISON_FIELDS = (
    "cover_psnr_db",
    "secret_clean_psnr_db",
    "secret_q50_psnr_db",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _style_axes(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=INK)
    axis.title.set_color(INK)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)


def _comparison_models(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    raw_models = comparison.get("models")
    if not isinstance(raw_models, list) or not 2 <= len(raw_models) <= 3:
        raise ValueError("comparison must contain two or three models")
    models: list[dict[str, Any]] = []
    for index, model in enumerate(raw_models):
        if not isinstance(model, dict):
            raise ValueError(f"comparison model {index} must be an object")
        name = model.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"comparison model {index} needs a non-empty name")
        for field in COMPARISON_FIELDS:
            try:
                value = float(model[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"comparison model {name!r} needs numeric field {field}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"comparison model {name!r} has non-finite {field}")
        models.append(model)
    return models


def _named_model(models: list[dict[str, Any]], name: str, role: str) -> dict[str, Any]:
    matches = [model for model in models if model["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"{role} model {name!r} must match exactly one comparison row")
    return matches[0]


def _metadata_name(comparison: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = comparison.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def comparison_context(
    comparison: dict[str, Any],
    models: list[dict[str, Any]] | None = None,
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    """Return a data-derived title, delta subtitle, candidate, and reference."""

    models = models or _comparison_models(comparison)
    selected_name = _metadata_name(
        comparison, ("selected_model_name", "selected_model", "candidate_model_name")
    )
    if selected_name is not None:
        selected = _named_model(models, selected_name, "selected")
    else:
        marked = [model for model in models if model.get("selected") is True]
        selected = (
            marked[0]
            if len(marked) == 1
            else max(models, key=lambda model: float(model["secret_clean_psnr_db"]))
        )

    reference_name = _metadata_name(
        comparison,
        ("upstream_model_name", "reference_model_name", "baseline_model_name"),
    )
    if reference_name is not None:
        reference = _named_model(models, reference_name, "reference")
    else:
        marked = [
            model
            for model in models
            if str(model.get("role", "")).lower() in {"upstream", "reference", "baseline"}
        ]
        reference = marked[0] if len(marked) == 1 else next(
            model for model in models if model is not selected
        )

    title_value = comparison.get("chart_title", comparison.get("description"))
    title = (
        title_value
        if isinstance(title_value, str) and title_value.strip()
        else "256×256 full-image steganography comparison"
    )
    cover_delta = float(selected["cover_psnr_db"]) - float(reference["cover_psnr_db"])
    secret_delta = float(selected["secret_clean_psnr_db"]) - float(
        reference["secret_clean_psnr_db"]
    )
    delta_subtitle = (
        f"{selected['name']} − {reference['name']}: "
        f"cover {cover_delta:+.2f} dB · clean secret {secret_delta:+.2f} dB"
    )
    return str(title), delta_subtitle, selected, reference


def plot_jpeg_sweep(benchmark: dict[str, Any], output: Path) -> None:
    metrics = benchmark.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("benchmark metrics must be an object")
    qualities = [50, 60, 70, 80, 90, 100]
    psnr_values = [float(metrics[f"jpeg_q{quality}_psnr_db"]) for quality in qualities[:-1]]
    psnr_values.append(float(metrics["secret_psnr_db"]))
    ssim_values = [float(metrics[f"jpeg_q{quality}_ssim"]) for quality in qualities[:-1]]
    ssim_values.append(float(metrics["secret_ssim"]))

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    figure.suptitle(
        "Secret-image recovery under real JPEG compression",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    for axis, values, label, color, limits in (
        (axes[0], psnr_values, "Secret PSNR (dB)", BLUE, (0.0, 40.0)),
        (axes[1], ssim_values, "Secret SSIM", ORANGE, (0.0, 1.0)),
    ):
        axis.plot(qualities, values, color=color, marker="o", linewidth=2.4, markersize=6)
        axis.fill_between(qualities, values, limits[0], color=color, alpha=0.08)
        axis.set_xlabel("Evaluation condition (Clean = uncompressed tensor path)")
        axis.set_ylabel(label)
        axis.set_xticks(qualities, ["50", "60", "70", "80", "90", "Clean"])
        axis.set_ylim(*limits)
        _style_axes(axis)
        for quality, value in zip(qualities, values, strict=True):
            axis.annotate(
                f"{value:.2f}" if label.endswith("(dB)") else f"{value:.3f}",
                (quality, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK,
            )
    evaluated = int(float(metrics.get("evaluated_images", 0)))
    cohort = f" · n={evaluated} held-out pairs" if evaluated else ""
    figure.text(
        0.5,
        -0.01,
        f"Source: results/benchmark_256.json{cohort} · Pillow/libjpeg 4:2:0",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_comparison(comparison: dict[str, Any], output: Path) -> None:
    models = _comparison_models(comparison)
    title, delta_subtitle, _, _ = comparison_context(comparison, models)
    positions = np.arange(len(COMPARISON_CATEGORIES))
    width = 0.78 / len(models)
    figure, axis = plt.subplots(figsize=(10.2, 5.7), constrained_layout=True)
    for index, model in enumerate(models):
        values = [float(model[field]) for field in COMPARISON_FIELDS]
        offset = (index - (len(models) - 1) / 2.0) * width
        bars = axis.bar(
            positions + offset,
            values,
            width * 0.94,
            label=str(model["name"]),
            color=MODEL_COLORS[index],
            alpha=0.9,
            edgecolor=INK,
            linewidth=0.5,
        )
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=8.5)
    axis.set_title(f"{title}\n{delta_subtitle}", fontsize=14, fontweight="bold")
    axis.set_ylabel("PSNR (dB)")
    axis.set_xticks(positions, COMPARISON_CATEGORIES)
    maximum = max(float(model[field]) for model in models for field in COMPARISON_FIELDS)
    axis.set_ylim(0, max(10.0, math.ceil((maximum + 4.0) / 5.0) * 5.0))
    axis.legend(frameon=False, loc="upper right")
    _style_axes(axis)
    evaluated = comparison.get("evaluation_images")
    cohort = f" · same n={int(evaluated)} held-out pairs" if evaluated is not None else ""
    figure.text(
        0.5,
        -0.01,
        f"Source: results/comparison_256.json{cohort}",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def paired_secret_psnr_evidence(
    paired: dict[str, Any],
) -> tuple[np.ndarray, float, float, float]:
    """Validate and return pair-level deltas, reported mean, and bootstrap CI."""

    raw_pairs = paired.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("paired comparison must contain at least one pair")
    deltas: list[float] = []
    for index, record in enumerate(raw_pairs):
        try:
            delta = float(
                record["delta_candidate_minus_baseline"]["secret_psnr_db"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"paired record {index} lacks a secret-PSNR delta") from error
        if not math.isfinite(delta):
            raise ValueError(f"paired record {index} has a non-finite secret-PSNR delta")
        deltas.append(delta)
    values = np.asarray(deltas, dtype=np.float64)
    try:
        summary = paired["secret_psnr_delta_db"]
        mean = float(summary["mean"])
        interval = summary["bootstrap_95_ci"]
        lower = float(interval["lower"])
        upper = float(interval["upper"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("paired comparison lacks its mean/bootstrap summary") from error
    if not np.isfinite((mean, lower, upper)).all():
        raise ValueError("paired comparison summary contains non-finite values")
    if not math.isclose(mean, float(values.mean()), abs_tol=1e-9):
        raise ValueError("reported paired mean does not match the per-pair deltas")
    if not lower <= mean <= upper:
        raise ValueError("paired bootstrap interval must contain the reported mean")
    declared_count = paired.get("pair_count")
    if declared_count is not None and int(declared_count) != len(values):
        raise ValueError("paired pair_count does not match the per-pair records")
    return values, mean, lower, upper


def plot_paired_comparison(paired: dict[str, Any], output: Path) -> None:
    deltas, mean, lower, upper = paired_secret_psnr_evidence(paired)
    positions = np.arange(1, len(deltas) + 1)
    figure, axis = plt.subplots(figsize=(10.2, 5.2), constrained_layout=True)
    axis.axhspan(
        lower,
        upper,
        color=BLUE,
        alpha=0.12,
        label="95% paired-bootstrap CI of mean",
        zorder=0,
    )
    axis.axhline(0.0, color=INK, linewidth=1.0, linestyle="--", label="No change", zorder=1)
    axis.scatter(
        positions,
        deltas,
        color=ORANGE,
        edgecolor=INK,
        linewidth=0.35,
        s=24,
        alpha=0.78,
        label="Per-pair delta",
        zorder=2,
    )
    axis.axhline(
        mean,
        color=BLUE,
        linewidth=2.2,
        label=f"Mean {mean:+.3f} dB",
        zorder=3,
    )
    axis.set_title(
        "Paired clean secret-recovery PSNR deltas\n"
        f"candidate − upstream · mean {mean:+.3f} dB · "
        f"95% bootstrap CI [{lower:+.3f}, {upper:+.3f}]",
        fontsize=13.5,
        fontweight="bold",
    )
    axis.set_xlabel("Held-out manifest pair (original order)")
    axis.set_ylabel("Secret PSNR delta (dB)")
    axis.set_xlim(0, len(deltas) + 1)
    axis.legend(frameon=False, loc="best")
    _style_axes(axis)
    figure.text(
        0.5,
        -0.01,
        f"Source: results/paired_final.json · n={len(deltas)} explicit cover/secret pairs",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _chart_map(
    benchmark_path: Path,
    comparison_path: Path,
    output_dir: Path,
    paired_path: Path | None,
) -> dict[str, Any]:
    charts: list[dict[str, Any]] = [
        {
            "artifact": (output_dir / "jpeg_robustness.png").as_posix(),
            "source": benchmark_path.as_posix(),
            "fields": [
                "metrics.secret_psnr_db",
                "metrics.secret_ssim",
                "metrics.jpeg_q{50,60,70,80,90}_{psnr_db,ssim}",
            ],
            "transformation": (
                "Direct ordered plot; clean uses a separate final categorical position"
            ),
            "grain": "One aggregate mean per codec condition over held-out cover/secret pairs",
        },
        {
            "artifact": (output_dir / "architecture_comparison.png").as_posix(),
            "source": comparison_path.as_posix(),
            "fields": [
                "models[].name",
                "models[].cover_psnr_db",
                "models[].secret_clean_psnr_db",
                "models[].secret_q50_psnr_db",
                "selected_model_name",
                "upstream_model_name",
            ],
            "transformation": "Grouped bars; title deltas are selected minus upstream/reference",
            "grain": "One aggregate mean per model and evaluation condition",
        },
    ]
    if paired_path is not None:
        charts.append(
            {
                "artifact": (output_dir / "paired_secret_psnr_delta.png").as_posix(),
                "source": paired_path.as_posix(),
                "fields": [
                    "pairs[].id",
                    "pairs[].delta_candidate_minus_baseline.secret_psnr_db",
                    "secret_psnr_delta_db.mean",
                    "secret_psnr_delta_db.bootstrap_95_ci.{lower,upper}",
                ],
                "transformation": (
                    "Manifest-order pair deltas with zero reference, observed mean, "
                    "and seeded paired-bootstrap 95% confidence interval"
                ),
                "grain": "One delta per explicit held-out cover/secret pair",
            }
        )
    return {"schema_version": 2, "charts": charts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("results/benchmark_256.json"))
    parser.add_argument("--comparison", type=Path, default=Path("results/comparison_256.json"))
    parser.add_argument("--paired", type=Path, default=Path("results/paired_final.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--chart-map", type=Path, default=Path("results/chart_map.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot_jpeg_sweep(_load(args.benchmark), args.output_dir / "jpeg_robustness.png")
    plot_comparison(_load(args.comparison), args.output_dir / "architecture_comparison.png")
    paired_path: Path | None = args.paired if args.paired.is_file() else None
    if paired_path is not None:
        plot_paired_comparison(
            _load(paired_path),
            args.output_dir / "paired_secret_psnr_delta.png",
        )
    chart_map = _chart_map(
        args.benchmark,
        args.comparison,
        args.output_dir,
        paired_path,
    )
    args.chart_map.parent.mkdir(parents=True, exist_ok=True)
    args.chart_map.write_text(
        json.dumps(chart_map, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
