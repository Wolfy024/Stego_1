import torch
from torch import nn
from Modules.ResidualBlock import ResidualBlock
from Modules.MultiScaleAttention import MultiScaleAttention
from Modules.FrequencyDecomposition import FrequencyDecomposition

class EncoderNetwork(nn.Module):
    """Encoder network for hiding secret image in cover image"""

    def __init__(self, input_channels=6, output_channels=3):
        super(EncoderNetwork, self).__init__()

        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        # Frequency decomposition module
        self.freq_decomp = FrequencyDecomposition(channels=128)

        # Multi-scale attention
        self.ms_attention = MultiScaleAttention(128)

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(6)])

        # Output layers
        self.output_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, cover, secret):
        # Concatenate cover and secret images
        x = torch.cat([cover, secret], dim=1)

        # Initial feature extraction
        features = self.initial_conv(x)

        # Adaptive frequency decomposition
        freq_features, band_weights = self.freq_decomp(features)

        # Multi-scale attention
        attended_features, attention_map = self.ms_attention(freq_features)

        # Residual processing
        processed_features = self.res_blocks(attended_features)

        # Generate stego image
        stego = self.output_conv(processed_features)

        # Add residual connection with cover image
        stego = cover + 0.1 * stego  # Small perturbation

        return stego, band_weights, attention_map