from torch import nn
from Modules.MultiScaleAttention import MultiScaleAttention

class ResidualBlock(nn.Module):
    """Enhanced residual block with attention"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.attention = MultiScaleAttention(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out, _ = self.attention(out)
        return self.relu(out + residual)
