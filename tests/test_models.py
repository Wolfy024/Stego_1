import unittest

import torch
from torch import nn

from stego.config import ModelConfig
from stego.invertible import CouplingAttention, HaarWavelet, InvertibleFlow, WaveletBandGate
from stego.models import CBAM, StegoGAN

torch.set_num_threads(1)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = StegoGAN(
            ModelConfig(
                base_channels=8,
                max_residual=0.03,
                target_cover_psnr=38.5,
                payload_bits=96,
                secret_size=None,
            )
        )

    def test_forward_shapes_and_residual_bound(self) -> None:
        cover = torch.rand(1, 3, 32, 32) * 2 - 1
        secret = torch.randint(0, 2, (1, 96)).float()
        outputs = self.model(cover, secret)
        self.assertEqual(outputs["stego"].shape, cover.shape)
        self.assertEqual(outputs["revealed_secret"].shape, (1, 96))
        self.assertLessEqual(outputs["residual"].abs().max().item(), 0.030001)
        rms = outputs["residual"].square().mean().sqrt().item()
        target_rms = 2.0 * 10.0 ** (-38.5 / 20.0)
        self.assertLessEqual(rms, target_rms + 1e-6)
        self.assertGreaterEqual(outputs["stego"].min().item(), -1.0)
        self.assertLessEqual(outputs["stego"].max().item(), 1.0)

    def test_architecture_uses_cbam_and_spectral_norm(self) -> None:
        self.assertTrue(any(isinstance(module, CBAM) for module in self.model.modules()))
        discriminator_modules = set(self.model.discriminator.modules())
        convolutions = [
            module
            for module in self.model.modules()
            if isinstance(module, nn.Conv2d) and module in discriminator_modules
        ]
        normalized = [
            module
            for module in convolutions
            if hasattr(module, "parametrizations") and "weight" in module.parametrizations
        ]
        self.assertEqual(len(normalized), len(convolutions))
        self.assertGreaterEqual(len(normalized), 5)
        self.assertEqual(self.model.bottleneck_channels, 64)

    def test_reveal_is_keyless(self) -> None:
        stego = torch.zeros(1, 3, 32, 32)
        revealed = self.model.reveal(stego)
        self.assertEqual(revealed.shape, (1, 96))

    def test_experimental_image_payload_mode(self) -> None:
        model = StegoGAN(ModelConfig(base_channels=8, payload_bits=None, secret_size=32))
        cover = torch.zeros(1, 3, 32, 32)
        outputs = model(cover, torch.ones_like(cover))
        self.assertEqual(outputs["revealed_secret"].shape, cover.shape)

    def test_haar_round_trip_is_exact(self) -> None:
        image = torch.randn(2, 3, 32, 32)
        wavelet = HaarWavelet()
        restored = wavelet(wavelet(image), reverse=True)
        self.assertTrue(torch.allclose(restored, image, atol=1e-6))

    def test_invertible_flow_round_trip_is_exact(self) -> None:
        flow = InvertibleFlow(blocks=2, growth_channels=8, clamp=1.0)
        values = torch.randn(1, 24, 8, 8)
        restored = flow(flow(values), reverse=True)
        self.assertTrue(torch.allclose(restored, values, atol=1e-5))

    def test_new_gates_start_as_identity(self) -> None:
        features = torch.randn(2, 16, 8, 8)
        residual = torch.randn(2, 12, 8, 8)
        self.assertTrue(torch.equal(CouplingAttention(16)(features), features))
        self.assertTrue(torch.equal(WaveletBandGate(12)(residual), residual))

    def test_wavelet_gated_model_uses_full_size_secret(self) -> None:
        config = ModelConfig(
            architecture="invertible",
            base_channels=8,
            payload_bits=None,
            secret_size=32,
            invertible_blocks=2,
            coupling_channels=8,
        )
        model = StegoGAN(config)
        cover = torch.rand(1, 3, 32, 32) * 2 - 1
        secret = torch.rand_like(cover) * 2 - 1
        outputs = model(cover, secret)
        self.assertEqual(outputs["stego"].shape, cover.shape)
        self.assertEqual(outputs["revealed_secret"].shape, secret.shape)


if __name__ == "__main__":
    unittest.main()
