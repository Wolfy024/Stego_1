from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from stego.data import discover_images


class EvaluationManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.coco = self.root / "coco"
        self.div2k = self.root / "div2k"
        self.coco.mkdir()
        self.div2k.mkdir()
        for index in range(24):
            (self.coco / f"coco_{index:03d}.png").write_bytes(f"coco-{index}".encode())
        for index in range(12):
            (self.div2k / f"div2k_{index:03d}.png").write_bytes(f"div2k-{index}".encode())
        self.script = Path(__file__).parents[1] / "scripts" / "create_eval_manifests.py"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, output: Path) -> None:
        subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--data-dir",
                str(self.coco),
                str(self.div2k),
                "--output-dir",
                str(output),
                "--excluded-images",
                "10",
                "--dev-pairs",
                "3",
                "--final-pairs",
                "4",
            ],
            check=True,
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _verify_manifest_digest(payload: dict[str, object]) -> None:
        expected = payload.pop("manifest_sha256")
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        assert expected == hashlib.sha256(canonical).hexdigest()

    def test_manifests_are_reproducible_hashed_and_disjoint(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        self._run(first)
        self._run(second)

        dev = json.loads((first / "dev.json").read_text(encoding="utf-8"))
        final = json.loads((first / "final.json").read_text(encoding="utf-8"))
        self.assertEqual(
            (first / "dev.json").read_bytes(),
            (second / "dev.json").read_bytes(),
        )
        self.assertEqual(
            (first / "final.json").read_bytes(),
            (second / "final.json").read_bytes(),
        )
        self.assertEqual(len(dev["pairs"]), 3)
        self.assertEqual(len(final["pairs"]), 4)
        self.assertEqual(
            dev["data_roots"],
            {"coco": "../coco", "div2k": "../div2k"},
        )

        def paths(payload: dict[str, object]) -> set[str]:
            pairs = payload["pairs"]
            assert isinstance(pairs, list)
            return {record[key] for record in pairs for key in ("cover_path", "secret_path")}

        dev_paths = paths(dev)
        final_paths = paths(final)
        self.assertEqual(len(dev_paths), 6)
        self.assertEqual(len(final_paths), 8)
        self.assertTrue(dev_paths.isdisjoint(final_paths))

        discovered = discover_images([self.coco, self.div2k])
        generator = torch.Generator().manual_seed(2026)
        excluded_indices = torch.randperm(len(discovered), generator=generator)[:10].tolist()
        excluded = {
            f"{path.parent.name}/{path.name}" for path in (discovered[i] for i in excluded_indices)
        }
        self.assertTrue((dev_paths | final_paths).isdisjoint(excluded))

        files = {
            f"{path.parent.name}/{path.name}": path
            for path in discover_images([self.coco, self.div2k])
        }
        for payload in (dev, final):
            for record in payload["pairs"]:
                for role in ("cover", "secret"):
                    image = files[record[f"{role}_path"]]
                    expected = hashlib.sha256(image.read_bytes()).hexdigest()
                    self.assertEqual(record[f"{role}_sha256"], expected)
            sidecar = first / f"{payload['cohort']}.json.sha256"
            expected_file_digest = hashlib.sha256(
                (first / f"{payload['cohort']}.json").read_bytes()
            ).hexdigest()
            self.assertEqual(sidecar.read_text().split()[0], expected_file_digest)
            self._verify_manifest_digest(dict(payload))

    def test_rejects_requests_larger_than_unused_pool(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--data-dir",
                str(self.coco),
                str(self.div2k),
                "--output-dir",
                str(self.root / "too-large"),
                "--excluded-images",
                "30",
                "--dev-pairs",
                "2",
                "--final-pairs",
                "2",
            ],
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unused images", completed.stderr)


if __name__ == "__main__":
    unittest.main()
