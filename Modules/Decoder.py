from torch import nn
from Modules.ResidualBlock import ResidualBlock
from Modules.MultiScaleAttention import MultiScaleAttention
from Modules.FrequencyDecomposition import FrequencyDecomposition

class DecoderNetwork(nn.Module):
    """Decoder network for extracting secret image from stego image"""

    def __init__(self, input_channels=3, output_channels=3):
        super(DecoderNetwork, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        # Frequency analysis for extraction
        self.freq_decomp = FrequencyDecomposition(channels=128)

        # Attention mechanism
        self.attention = MultiScaleAttention(128)

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(4)])

        # Output layers
        self.output_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, stego):
        features = self.conv_blocks(stego)

        # Frequency analysis
        freq_features, _ = self.freq_decomp(features)

        # Attention
        attended_features, _ = self.attention(freq_features)

        # Residual processing
        processed_features = self.res_blocks(attended_features)

        # Extract secret
        secret = self.output_conv(processed_features)

        return secret