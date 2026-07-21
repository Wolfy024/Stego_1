"""Create README-ready charts from committed benchmark evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2563EB"
ORANGE = "#EA580C"
GREEN = "#15803D"
INK = "#172033"
GRID = "#D7DEE8"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _style_axes(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=INK)
    axis.title.set_color(INK)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)


def plot_jpeg_sweep(benchmark: dict[str, object], output: Path) -> None:
    metrics = benchmark["metrics"]
    assert isinstance(metrics, dict)
    qualities = [50, 60, 70, 80, 90, 100]
    psnr_values = [float(metrics[f"jpeg_q{quality}_psnr_db"]) for quality in qualities[:-1]]
    psnr_values.append(float(metrics["secret_psnr_db"]))
    ssim_values = [float(metrics[f"jpeg_q{quality}_ssim"]) for quality in qualities[:-1]]
    ssim_values.append(float(metrics["secret_ssim"]))

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    figure.suptitle(
        "High clean fidelity, but JPEG destroys the hidden image",
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
        axis.set_xlabel("JPEG quality factor (100 = no compression)")
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
    figure.text(
        0.5,
        -0.01,
        "Source: results/benchmark_256.json · n=200 held-out pairs · Pillow/libjpeg 4:2:0",
        ha="center",
        fontsize=8.5,
        color="#526173",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_comparison(comparison: dict[str, object], output: Path) -> None:
    models = comparison["models"]
    assert isinstance(models, list)
    categories = ["Cover → stego", "Secret (clean)", "Secret (QF-50)"]
    fields = ["cover_psnr_db", "secret_clean_psnr_db", "secret_q50_psnr_db"]
    positions = np.arange(len(categories))
    width = 0.34
    figure, axis = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    for index, (model, color) in enumerate(zip(models, (ORANGE, BLUE), strict=True)):
        values = [float(model[field]) for field in fields]
        bars = axis.bar(
            positions + (index - 0.5) * width,
            values,
            width,
            label=str(model["name"]),
            color=color,
            alpha=0.9,
        )
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=9)
    axis.set_title(
        "Invertibility adds +20.19 dB clean secret recovery",
        fontsize=16,
        fontweight="bold",
    )
    axis.set_ylabel("PSNR (dB)")
    axis.set_xticks(positions, categories)
    axis.set_ylim(0, 43)
    axis.legend(frameon=False, loc="upper right")
    _style_axes(axis)
    figure.text(
        0.5,
        -0.01,
        "Source: results/comparison_256.json · same n=200 held-out pairs",
        ha="center",
        fontsize=8.5,
        color="#526173",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("results/benchmark_256.json"))
    parser.add_argument("--comparison", type=Path, default=Path("results/comparison_256.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    args = parser.parse_args()

    plot_jpeg_sweep(_load(args.benchmark), args.output_dir / "jpeg_robustness.png")
    plot_comparison(_load(args.comparison), args.output_dir / "architecture_comparison.png")
    chart_map = {
        "schema_version": 1,
        "charts": [
            {
                "artifact": "assets/jpeg_robustness.png",
                "source": "results/benchmark_256.json",
                "fields": ["metrics.secret_psnr_db", "metrics.secret_ssim", "metrics.jpeg_q*"],
                "transformation": "Direct ordered plot; clean is displayed as QF=100",
                "grain": "Mean over 200 held-out cover/secret pairs",
            },
            {
                "artifact": "assets/architecture_comparison.png",
                "source": "results/comparison_256.json",
                "fields": [
                    "cover_psnr_db",
                    "secret_clean_psnr_db",
                    "secret_q50_psnr_db",
                ],
                "transformation": "Grouped bars; no normalization",
                "grain": "One bar per model and evaluation condition",
            },
        ],
    }
    Path("results/chart_map.json").write_text(
        json.dumps(chart_map, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
