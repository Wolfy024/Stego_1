from torchvision.models import vgg19
from torch import nn
from torch.nn import functional as F
import torch


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG19"""

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Fix the deprecated 'pretrained' parameter
        from torchvision.models import VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

        self.vgg_layers = nn.ModuleList([
            vgg[:4],  # relu1_2
            vgg[:9],  # relu2_2
            vgg[:18],  # relu3_4
            vgg[:27],  # relu4_4
        ])

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        # Normalization for ImageNet pretrained models
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_input(self, x):
        """Normalize input for VGG (expects ImageNet normalization)"""
        # Assume input is in [-1, 1], convert to [0, 1]
        x = (x + 1) / 2
        # Apply ImageNet normalization
        return (x - self.mean) / self.std

    def forward(self, x, y):
        # Ensure inputs have 3 channels (RGB)
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {x.shape[1]} channels")
        if y.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {y.shape[1]} channels")

        # Normalize inputs for VGG
        x_norm = self.normalize_input(x)
        y_norm = self.normalize_input(y)

        loss = 0

        # Process through each VGG layer separately, starting from original input each time
        for vgg_layer in self.vgg_layers:
            x_feat = vgg_layer(x_norm)  # Always start from original input
            y_feat = vgg_layer(y_norm)  # Always start from original input
            loss += F.mse_loss(x_feat, y_feat)

        return loss