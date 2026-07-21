import unittest

import torch

from stego.jpeg import DifferentiableJPEG


class DifferentiableJPEGTests(unittest.TestCase):
    def test_shape_range_and_gradient(self) -> None:
        image = (torch.rand(2, 3, 35, 41) * 2 - 1).requires_grad_()
        output = DifferentiableJPEG(quality=50)(image)
        self.assertEqual(output.shape, image.shape)
        self.assertGreaterEqual(output.min().item(), -1.0)
        self.assertLessEqual(output.max().item(), 1.0)
        output.square().mean().backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(image.grad.abs().sum().item(), 0.0)

    def test_quality_validation(self) -> None:
        with self.assertRaises(ValueError):
            DifferentiableJPEG(quality=0)

    def test_chroma_subsampling_changes_color_detail(self) -> None:
        checkerboard = torch.zeros(1, 3, 32, 32)
        checkerboard[:, 0, ::2, ::2] = 1.0
        checkerboard[:, 2, 1::2, 1::2] = 1.0
        with_subsampling = DifferentiableJPEG(quality=90, chroma_subsampling=True)(checkerboard)
        without_subsampling = DifferentiableJPEG(quality=90, chroma_subsampling=False)(checkerboard)
        self.assertGreater(
            (with_subsampling - without_subsampling).abs().mean().item(),
            0.01,
        )


if __name__ == "__main__":
    unittest.main()
