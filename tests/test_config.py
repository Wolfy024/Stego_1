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


if __name__ == "__main__":
    unittest.main()
