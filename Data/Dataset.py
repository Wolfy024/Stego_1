import os
import random
from PIL import Image
from torch.utils.data import Dataset

class SteganographyDataset(Dataset):
    """Dataset class for steganography training from one main folder"""

    def __init__(self, root_dir, transform=None, image_size=256, max_images=6000):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.max_images = max_images

        # Recursively collect all image paths
        self.images = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.images.append(os.path.join(root, f))

        if not self.images:
            raise RuntimeError(f"No images found in {root_dir}")

        # Limit the number of images if max_images is specified
        if self.max_images is not None and len(self.images) > self.max_images:
            # Shuffle to get random subset
            random.shuffle(self.images)
            self.images = self.images[:self.max_images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Cover image = sequential
        cover_path = self.images[idx]
        cover = Image.open(cover_path).convert('RGB')
        cover = cover.resize((self.image_size, self.image_size))

        # Secret image = random
        secret_path = random.choice(self.images)
        secret = Image.open(secret_path).convert('RGB')
        secret = secret.resize((self.image_size, self.image_size))

        if self.transform:
            cover = self.transform(cover)
            secret = self.transform(secret)

        return cover, secret