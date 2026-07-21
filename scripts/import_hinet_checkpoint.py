"""Import official HiNet dense-coupling weights into the local gated variant."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from stego.config import ExperimentConfig
from stego.models import StegoGAN
from stego.training import Trainer, seed_everything

SUBNET_NAMES = {"r": "rho", "y": "eta", "f": "phi"}
EXPECTED_SOURCE_SHA256 = "d89a83e0e549e9bddde301d631b8614cc724c8d9f11feaf2221791aa02c122a7"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/invertible_h100.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/invertible_warmstart"))
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    if config.model.architecture != "invertible":
        raise ValueError("checkpoint import requires an invertible model config")
    if config.model.invertible_blocks != 16 or config.model.coupling_channels != 32:
        raise ValueError("official HiNet weights require 16 blocks and 32 growth channels")

    source_sha256 = _sha256(args.source)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "source checkpoint SHA-256 does not match the disclosed official HiNet file: "
            f"{source_sha256}"
        )

    seed_everything(config.data.seed)
    payload = torch.load(args.source, map_location="cpu", weights_only=True)
    source = payload["net"]
    model = StegoGAN(config.model)
    target = model.state_dict()
    imported = 0
    for block_index in range(16):
        for source_subnet, target_subnet in SUBNET_NAMES.items():
            for convolution in range(1, 6):
                source_prefix = (
                    f"module.model.inv{block_index + 1}.{source_subnet}.conv{convolution}"
                )
                if convolution < 5:
                    target_prefix = (
                        f"encoder.flow.blocks.{block_index}.{target_subnet}."
                        f"layers.{convolution - 1}"
                    )
                else:
                    target_prefix = (
                        f"encoder.flow.blocks.{block_index}.{target_subnet}.output"
                    )
                for suffix in ("weight", "bias"):
                    source_key = f"{source_prefix}.{suffix}"
                    target_key = f"{target_prefix}.{suffix}"
                    if source[source_key].shape != target[target_key].shape:
                        raise ValueError(
                            f"shape mismatch for {source_key}: "
                            f"{source[source_key].shape} vs {target[target_key].shape}"
                        )
                    target[target_key] = source[source_key]
                    imported += 1
    model.load_state_dict(target)

    trainer = Trainer(model, config, args.output_dir, torch.device("cpu"))
    checkpoint = trainer.save_checkpoint("latest.pt")
    print(f"imported_tensors={imported}")
    print(f"checkpoint={checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
