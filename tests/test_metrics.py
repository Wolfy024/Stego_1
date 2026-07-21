import unittest

import torch

from stego.metrics import bit_accuracy, extraction_similarity, psnr, ssim


class MetricTests(unittest.TestCase):
    def test_identical_images(self) -> None:
        image = torch.rand(2, 3, 16, 16) * 2 - 1
        self.assertTrue(torch.isinf(psnr(image, image)).all())
        self.assertTrue(torch.allclose(ssim(image, image), torch.ones(2), atol=1e-5))
        self.assertTrue(
            torch.allclose(extraction_similarity(image, image), torch.full((2,), 100.0))
        )

    def test_psnr_uses_minus_one_to_one_range(self) -> None:
        reference = torch.zeros(1, 3, 16, 16)
        estimate = torch.ones_like(reference) * 0.1
        self.assertAlmostEqual(psnr(reference, estimate).item(), 26.0206, places=3)

    def test_exact_bit_accuracy(self) -> None:
        payload = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        logits = torch.tensor([[-2.0, 3.0, -1.0, -4.0]])
        self.assertEqual(bit_accuracy(payload, logits).item(), 75.0)


if __name__ == "__main__":
    unittest.main()
