import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from stego.cli import _apply_train_overrides
from stego.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from stego.invertible import AttentiveDenseSubnet
from stego.models import StegoGAN
from stego.training import Trainer


def _invertible_config(*, router: bool, router_only: bool = False) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            architecture="invertible",
            base_channels=8,
            payload_bits=None,
            secret_size=32,
            invertible_blocks=1,
            coupling_channels=8,
            coupling_clamp=1.0,
            orthogonal_router=router,
        ),
        data=DataConfig(image_size=32, workers=0),
        train=TrainConfig(amp=False, router_only=router_only),
    )


class RouterIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)

    def test_router_insertion_preserves_trained_model_outputs_exactly(self) -> None:
        baseline = StegoGAN(_invertible_config(router=False).model).eval()
        for module in baseline.modules():
            if isinstance(module, AttentiveDenseSubnet):
                torch.nn.init.normal_(module.output.weight, std=0.01)
                torch.nn.init.normal_(module.output.bias, std=0.01)

        routed = StegoGAN(_invertible_config(router=True).model).eval()
        incompatible = routed.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(".router." in key for key in incompatible.missing_keys))

        cover = torch.rand(2, 3, 32, 32) * 2.0 - 1.0
        secret = torch.rand_like(cover) * 2.0 - 1.0
        baseline_outputs = baseline(cover, secret)
        routed_outputs = routed(cover, secret)

        for key in ("stego", "latent", "revealed_secret"):
            self.assertTrue(torch.equal(baseline_outputs[key], routed_outputs[key]), key)

    def test_router_only_freezes_every_other_generator_parameter(self) -> None:
        config = _invertible_config(router=True, router_only=True)
        config.validate()
        with tempfile.TemporaryDirectory() as output_dir:
            trainer = Trainer(StegoGAN(config.model), config, output_dir, torch.device("cpu"))

        named_parameters = dict(trainer.model.encoder.named_parameters())
        router_names = [name for name in named_parameters if ".router." in name]
        self.assertTrue(router_names)
        self.assertTrue(all(named_parameters[name].requires_grad for name in router_names))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in named_parameters.items()
                if name not in router_names
            )
        )

    def test_router_only_requires_router_and_is_mutually_exclusive(self) -> None:
        missing_router = _invertible_config(router=False, router_only=True)
        with self.assertRaisesRegex(ValueError, "requires orthogonal_router"):
            missing_router.validate()

        conflicting = _invertible_config(router=True, router_only=True)
        conflicting.model.invertible_blocks = 16
        conflicting.train.gates_only = True
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            conflicting.validate()

    def test_warm_start_allows_only_new_router_keys_but_strict_load_rejects_them(self) -> None:
        old_config = _invertible_config(router=False)
        new_config = _invertible_config(router=True, router_only=True)
        old_config.validate()
        new_config.validate()

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            old_trainer = Trainer(
                StegoGAN(old_config.model),
                old_config,
                output_path / "old",
                torch.device("cpu"),
            )
            checkpoint = old_trainer.save_checkpoint()
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload["config"]["model"].pop("orthogonal_router")
            torch.save(payload, checkpoint)

            warm_trainer = Trainer(
                StegoGAN(new_config.model),
                new_config,
                output_path / "warm",
                torch.device("cpu"),
            )
            warm_trainer.load_checkpoint(
                checkpoint,
                resume_optimizers=False,
                resume_state=False,
                allow_new_adapters=True,
            )

            strict_trainer = Trainer(
                StegoGAN(new_config.model),
                new_config,
                output_path / "strict",
                torch.device("cpu"),
            )
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                strict_trainer.load_checkpoint(
                    checkpoint,
                    resume_optimizers=False,
                    resume_state=False,
                )

            corrupt_checkpoint = output_path / "missing_regular_weight.pt"
            regular_key = next(iter(payload["model"]))
            payload["model"].pop(regular_key)
            torch.save(payload, corrupt_checkpoint)
            with self.assertRaisesRegex(RuntimeError, "incompatible warm start"):
                warm_trainer.load_checkpoint(
                    corrupt_checkpoint,
                    resume_optimizers=False,
                    resume_state=False,
                    allow_new_adapters=True,
                )

    def test_router_only_cli_override_clears_inherited_gate_mode(self) -> None:
        config = _invertible_config(router=False)
        config.train.gates_only = True
        args = argparse.Namespace(
            gates_only=False,
            router_only=True,
            orthogonal_router=True,
        )

        _apply_train_overrides(config, args)

        self.assertTrue(config.model.orthogonal_router)
        self.assertTrue(config.train.router_only)
        self.assertFalse(config.train.gates_only)


if __name__ == "__main__":
    unittest.main()
