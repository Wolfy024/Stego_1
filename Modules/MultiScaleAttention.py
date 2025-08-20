import torch
from torch import nn
from torch.nn import functional as F

class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for preserving perceptually important regions"""

    def __init__(self, channels, scales=[1, 2, 4]):
        super(MultiScaleAttention, self).__init__()
        self.scales = scales
        self.attention_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid()
            ) for _ in scales
        ])

        self.fusion_conv = nn.Conv2d(len(scales), 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        attention_maps = []

        for i, scale in enumerate(self.scales):
            if scale > 1:
                # Downsample
                x_scaled = F.avg_pool2d(x, scale, scale)
                attention = self.attention_convs[i](x_scaled)
                # Upsample back
                attention = F.interpolate(attention, size=(H, W), mode='bilinear', align_corners=False)
            else:
                attention = self.attention_convs[i](x)

            attention_maps.append(attention)

        # Fuse multi-scale attention maps
        combined_attention = torch.cat(attention_maps, dim=1)
        final_attention = self.sigmoid(self.fusion_conv(combined_attention))

        return x * final_attention, final_attention

