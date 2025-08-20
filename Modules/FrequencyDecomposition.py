import torch
from torch import nn


class FrequencyDecomposition(nn.Module):
    """Simplified frequency decomposition module to avoid FFT dimension issues"""

    def __init__(self, channels=3, num_bands=8):
        super(FrequencyDecomposition, self).__init__()
        self.num_bands = num_bands
        self.channels = channels

        # Use learnable convolutions to simulate frequency decomposition
        self.freq_conv = nn.Conv2d(channels, channels * num_bands, 3, padding=1)
        self.band_weights = nn.Parameter(torch.ones(num_bands))

        # Band selection network
        self.band_selector = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_bands),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Apply learnable frequency filters (simulating frequency decomposition)
        freq_features = self.freq_conv(x)  # [B, C * num_bands, H, W]
        freq_features = freq_features.view(B, C, self.num_bands, H, W)

        # Adaptive band selection
        band_scores = self.band_selector(x)  # [B, num_bands]
        band_scores = band_scores.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, num_bands, 1, 1]

        # Weighted combination of frequency bands
        selected_bands = (freq_features * band_scores).sum(dim=2)  # [B, C, H, W]

        return selected_bands, band_scores.squeeze()