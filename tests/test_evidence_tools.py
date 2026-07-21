from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import torch


def _import_script(name: str) -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLOT_RESULTS = _import_script("plot_results")
VALIDATE_RESULTS = _import_script("validate_results")


def _models() -> list[dict[str, object]]:
    return [
        {
            "name": "Wavelet-gated INN",
            "cover_psnr_db": 38.6037,
            "cover_ssim": 0.9635,
            "secret_clean_psnr_db": 36.9553,
            "secret_clean_ssim": 0.9658,
            "secret_q50_psnr_db": 11.2268,
            "secret_q50_ssim": 0.0093,
        },
        {
            "name": "CBAM U-Net",
            "cover_psnr_db": 38.6034,
            "cover_ssim": 0.9737,
            "secret_clean_psnr_db": 16.1546,
            "secret_clean_ssim": 0.3792,
            "secret_q50_psnr_db": 15.1110,
            "secret_q50_ssim": 0.2668,
        },
        {
            "name": "Upstream HiNet",
            "cover_psnr_db": 38.5836,
            "cover_ssim": 0.9632,
            "secret_clean_psnr_db": 36.5823,
            "secret_clean_ssim": 0.9638,
            "secret_q50_psnr_db": 11.2200,
            "secret_q50_ssim": 0.0090,
        },
    ]


def _comparison() -> dict[str, object]:
    return {
        "description": "Same-split 256×256 comparison",
        "evaluation_images": 200,
        "selected_model_name": "Wavelet-gated INN",
        "upstream_model_name": "Upstream HiNet",
        "models": _models(),
    }


def _benchmark() -> dict[str, object]:
    selected = _models()[0]
    return {
        "evaluation_precision": "FP32 (autocast disabled)",
        "metrics": {
            "cover_psnr_db": selected["cover_psnr_db"],
            "cover_ssim": selected["cover_ssim"],
            "secret_psnr_db": selected["secret_clean_psnr_db"],
            "secret_ssim": selected["secret_clean_ssim"],
            "jpeg_q50_psnr_db": selected["secret_q50_psnr_db"],
            "jpeg_q50_ssim": selected["secret_q50_ssim"],
            "jpeg_q60_psnr_db": 11.23,
            "jpeg_q60_ssim": 0.0095,
            "jpeg_q70_psnr_db": 11.24,
            "jpeg_q70_ssim": 0.0097,
            "jpeg_q80_psnr_db": 11.25,
            "jpeg_q80_ssim": 0.0099,
            "jpeg_q90_psnr_db": 11.30,
            "jpeg_q90_ssim": 0.0123,
            "evaluated_images": 200,
        },
    }


def _paired() -> dict[str, object]:
    deltas = [0.2, 0.4, -0.1]
    mean = sum(deltas) / len(deltas)
    pairs = [
        {
            "id": f"pair-{index}",
            "delta_candidate_minus_baseline": {
                "secret_psnr_db": delta,
            },
        }
        for index, delta in enumerate(deltas)
    ]
    return {
        "comparison_direction": "candidate minus baseline; positive is better",
        "pair_count": len(pairs),
        "evaluation_protocol": {
            "precision": "float32 (autocast disabled, TF32 disabled)",
        },
        "aggregate_metrics": {
            "candidate_minus_baseline": {
                "cover_psnr_db": {"mean": 0.02},
                "secret_psnr_db": {"mean": mean},
            }
        },
        "secret_psnr_delta_db": {
            "mean": mean,
            "bootstrap_95_ci": {
                "lower": -0.05,
                "upper": 0.4,
            },
        },
        "pairs": pairs,
    }


class PlotResultsTests(unittest.TestCase):
    def test_three_model_title_and_deltas_are_data_derived(self) -> None:
        comparison = _comparison()

        title, subtitle, selected, upstream = PLOT_RESULTS.comparison_context(comparison)

        self.assertEqual(title, comparison["description"])
        self.assertEqual(selected["name"], "Wavelet-gated INN")
        self.assertEqual(upstream["name"], "Upstream HiNet")
        self.assertIn("cover +0.02 dB", subtitle)
        self.assertIn("clean secret +0.37 dB", subtitle)
        self.assertNotIn("+20.19", subtitle)

    def test_comparison_plot_accepts_three_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.png"

            PLOT_RESULTS.plot_comparison(_comparison(), output)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1_000)

    def test_plot_main_adds_optional_paired_chart_and_traceable_chart_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_path = root / "benchmark.json"
            comparison_path = root / "comparison.json"
            paired_path = root / "paired.json"
            output_dir = root / "assets"
            chart_map_path = root / "chart_map.json"
            benchmark_path.write_text(json.dumps(_benchmark()), encoding="utf-8")
            comparison_path.write_text(json.dumps(_comparison()), encoding="utf-8")
            paired_path.write_text(json.dumps(_paired()), encoding="utf-8")

            result = PLOT_RESULTS.main(
                [
                    "--benchmark",
                    str(benchmark_path),
                    "--comparison",
                    str(comparison_path),
                    "--paired",
                    str(paired_path),
                    "--output-dir",
                    str(output_dir),
                    "--chart-map",
                    str(chart_map_path),
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "jpeg_robustness.png").is_file())
            self.assertTrue((output_dir / "architecture_comparison.png").is_file())
            self.assertTrue((output_dir / "paired_secret_psnr_delta.png").is_file())
            chart_map = json.loads(chart_map_path.read_text(encoding="utf-8"))
            paired_chart = chart_map["charts"][2]
            self.assertEqual(paired_chart["source"], paired_path.as_posix())
            self.assertIn(
                "pairs[].delta_candidate_minus_baseline.secret_psnr_db",
                paired_chart["fields"],
            )
            self.assertEqual(
                paired_chart["grain"],
                "One delta per explicit held-out cover/secret pair",
            )


class ValidateResultsTests(unittest.TestCase):
    def test_selected_model_is_resolved_by_name_not_position(self) -> None:
        comparison = _comparison()

        selected = VALIDATE_RESULTS.resolve_selected_model(_benchmark(), comparison)
        upstream = VALIDATE_RESULTS.resolve_upstream_model(comparison)

        self.assertEqual(selected["name"], "Wavelet-gated INN")
        self.assertEqual(upstream["name"], "Upstream HiNet")
        self.assertIs(selected, comparison["models"][0])

    def test_unique_metric_reconciliation_can_resolve_legacy_evidence(self) -> None:
        comparison = {"models": _models()}
        benchmark = _benchmark()

        selected = VALIDATE_RESULTS.resolve_selected_model(benchmark, comparison)

        self.assertEqual(selected["name"], "Wavelet-gated INN")

    def test_fp32_declaration_rejects_reduced_or_mixed_precision(self) -> None:
        self.assertTrue(
            VALIDATE_RESULTS.is_explicit_fp32(
                VALIDATE_RESULTS.evaluation_precision(_benchmark())
            )
        )
        self.assertFalse(VALIDATE_RESULTS.is_explicit_fp32("BF16 autocast"))
        self.assertFalse(VALIDATE_RESULTS.is_explicit_fp32("mixed FP32/BF16"))
        self.assertFalse(VALIDATE_RESULTS.is_explicit_fp32(None))

    def test_candidate_and_paired_pareto_deltas_are_strictly_positive(self) -> None:
        comparison = _comparison()
        candidate = VALIDATE_RESULTS.resolve_selected_model(_benchmark(), comparison)
        upstream = VALIDATE_RESULTS.resolve_upstream_model(comparison)
        self.assertIsNotNone(upstream)

        model_deltas = VALIDATE_RESULTS.model_pareto_deltas(candidate, upstream)
        paired_deltas = VALIDATE_RESULTS.paired_pareto_deltas(_paired())

        self.assertTrue(all(delta > 0.0 for delta in model_deltas.values()))
        self.assertTrue(all(delta > 0.0 for delta in paired_deltas.values()))

    def test_paired_integrity_recomputes_mean_from_pair_grain(self) -> None:
        valid, evidence = VALIDATE_RESULTS.paired_evidence_integrity(_paired())

        self.assertTrue(valid)
        self.assertIn("n=3", evidence)

        corrupted = _paired()
        corrupted["secret_psnr_delta_db"]["mean"] = 99.0
        valid, _ = VALIDATE_RESULTS.paired_evidence_integrity(corrupted)
        self.assertFalse(valid)

    def test_adapter_change_count_supports_gates_and_new_router_outputs(self) -> None:
        warm = {
            "flow.0.band_gate.weight": torch.zeros(2),
        }
        selected = {
            "flow.0.band_gate.weight": torch.tensor([0.0, 1.0]),
            "flow.0.router.skew_predictor.0.weight": torch.ones(4),
            "flow.0.router.skew_predictor.2.weight": torch.tensor([0.0, 2.0]),
            "flow.0.router.skew_predictor.2.bias": torch.tensor([3.0]),
        }

        changed = VALIDATE_RESULTS.changed_adapter_parameter_count(warm, selected)

        self.assertEqual(changed, 3)


if __name__ == "__main__":
    unittest.main()
