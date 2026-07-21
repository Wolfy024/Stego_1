import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from stego.data import ImagePairDataset, discover_images, split_images


class DatasetTests(unittest.TestCase):
    def test_disjoint_split_and_deterministic_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(10):
                pixels = np.full((40, 50, 3), index * 20, dtype=np.uint8)
                Image.fromarray(pixels).save(root / f"{index:02d}.png")
            images = discover_images([root])
            train, validation = split_images(images, 0.8, seed=42)
            self.assertFalse(set(train) & set(validation))
            dataset = ImagePairDataset(train, 32, augment=False, seed=42, payload_bits=96)
            first = dataset[0]
            second = dataset[0]
            self.assertTrue(first[0].equal(second[0]))
            self.assertTrue(first[1].equal(second[1]))
            self.assertEqual(first[0].shape, (3, 32, 32))
            self.assertEqual(first[1].shape, (96,))


if __name__ == "__main__":
    unittest.main()
