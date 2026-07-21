import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from stego.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from stego.invertible import WaveletGatedInvertibleHider
from stego.models import StegoGAN
from stego.training import Trainer

torch.set_num_threads(1)


def _invertible_config(*, orthogonal_router: bool = False) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            architecture="invertible",
            base_channels=8,
            payload_bits=None,
            secret_size=32,
            invertible_blocks=1,
            coupling_channels=8,
            orthogonal_router=orthogonal_router,
        ),
        data=DataConfig(image_size=32, workers=0),
        train=TrainConfig(batch_size=1, amp=False),
    )


def _trainer(config: ExperimentConfig, output_dir: Path) -> Trainer:
    return Trainer(StegoGAN(config.model), config, output_dir, torch.device("cpu"))


class TrainingRegressionTests(unittest.TestCase):
    def test_fixed_latent_is_invariant_to_batch_position_and_size(self) -> None:
        hider = WaveletGatedInvertibleHider(
            blocks=1,
            growth_channels=8,
            latent_noise_std=1.0,
            latent_seed=17,
        ).eval()
        single = hider._fixed_latent(torch.empty(1, 12, 16, 16))
        batch = hider._fixed_latent(torch.empty(4, 12, 16, 16))

        for position in range(batch.shape[0]):
            self.assertTrue(torch.equal(single[0], batch[position]))

    def test_clean_only_evaluation_emits_no_jpeg_metrics(self) -> None:
        torch.manual_seed(3)
        config = _invertible_config()
        cover = torch.rand(2, 3, 32, 32) * 2.0 - 1.0
        secret = torch.rand_like(cover) * 2.0 - 1.0
        loader = DataLoader(TensorDataset(cover, secret), batch_size=1)

        with tempfile.TemporaryDirectory() as directory:
            metrics = _trainer(config, Path(directory)).evaluate(
                loader,
                jpeg_qualities=(),
                real_codec=True,
            )

        self.assertIn("cover_psnr_db", metrics)
        self.assertIn("secret_psnr_db", metrics)
        self.assertEqual(metrics["evaluated_images"], 2.0)
        self.assertFalse(any(name.startswith("jpeg_") for name in metrics))

    def test_warm_start_accepts_only_expected_new_adapter_keys(self) -> None:
        torch.manual_seed(5)
        source_config = _invertible_config()
        target_config = _invertible_config(orthogonal_router=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _trainer(source_config, root / "source")
            checkpoint = source.save_checkpoint()
            target = _trainer(target_config, root / "target")

            target.load_checkpoint(
                checkpoint,
                resume_optimizers=False,
                resume_state=False,
                allow_new_adapters=True,
            )

            source_state = source.model.state_dict()
            target_state = target.model.state_dict()
            shared_keys = source_state.keys() & target_state.keys()
            self.assertTrue(shared_keys)
            for key in shared_keys:
                self.assertTrue(torch.equal(source_state[key], target_state[key]), key)
            new_keys = target_state.keys() - source_state.keys()
            self.assertTrue(new_keys)
            self.assertTrue(
                all(
                    key.startswith("encoder.flow.blocks.") and ".router." in key
                    for key in new_keys
                )
            )

    def test_warm_start_rejects_corrupt_non_adapter_state(self) -> None:
        source_config = _invertible_config()
        target_config = _invertible_config(orthogonal_router=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = _trainer(source_config, root / "source").save_checkpoint()
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            removed_key = next(
                key for key in payload["model"] if key.startswith("encoder.flow.")
            )
            del payload["model"][removed_key]
            corrupt_checkpoint = root / "corrupt.pt"
            torch.save(payload, corrupt_checkpoint)

            target = _trainer(target_config, root / "target")
            with self.assertRaisesRegex(RuntimeError, "incompatible warm start"):
                target.load_checkpoint(
                    corrupt_checkpoint,
                    resume_optimizers=False,
                    resume_state=False,
                    allow_new_adapters=True,
                )

    def test_strict_resume_rejects_architecture_evolution(self) -> None:
        source_config = _invertible_config()
        target_config = _invertible_config(orthogonal_router=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = _trainer(source_config, root / "source").save_checkpoint()
            target = _trainer(target_config, root / "target")

            with self.assertRaises(RuntimeError):
                target.load_checkpoint(
                    checkpoint,
                    resume_optimizers=False,
                    resume_state=False,
                )


if __name__ == "__main__":
    unittest.main()
