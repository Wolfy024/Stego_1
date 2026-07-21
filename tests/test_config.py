import unittest

from stego.config import DataConfig, ExperimentConfig, LossConfig, ModelConfig, TrainConfig


class ConfigTests(unittest.TestCase):
    def test_negative_loss_weight_is_rejected(self) -> None:
        config = ExperimentConfig(loss=LossConfig(secret=-1.0))
        with self.assertRaisesRegex(ValueError, "secret"):
            config.validate()

    def test_invertible_payload_must_match_image_size(self) -> None:
        config = ExperimentConfig(
            model=ModelConfig(architecture="invertible", secret_size=128),
            data=DataConfig(image_size=256),
        )
        with self.assertRaisesRegex(ValueError, "secret_size"):
            config.validate()

    def test_nonpositive_step_limit_is_rejected(self) -> None:
        config = ExperimentConfig(train=TrainConfig(max_train_steps=0))
        with self.assertRaisesRegex(ValueError, "max_train_steps"):
            config.validate()

    def test_removed_disabled_experiment_fields_are_checkpoint_compatible(self) -> None:
        raw = ExperimentConfig(
            model=ModelConfig(
                architecture="invertible",
                secret_size=256,
                invertible_blocks=16,
            ),
            data=DataConfig(image_size=256),
        ).to_dict()
        raw["model"].update(
            {
                "latent_calibration": False,
                "latent_predictor_channels": 0,
                "latent_predictor_blocks": 3,
            }
        )
        raw["loss"]["latent_consistency"] = 0.0
        raw["train"]["latent_only"] = False

        restored = ExperimentConfig.from_dict(raw)

        self.assertEqual(restored.model.architecture, "invertible")

    def test_removed_active_experiment_fields_are_rejected(self) -> None:
        raw = ExperimentConfig().to_dict()
        raw["model"]["latent_calibration"] = True

        with self.assertRaisesRegex(ValueError, "removed experimental option"):
            ExperimentConfig.from_dict(raw)

if __name__ == "__main__":
    unittest.main()
