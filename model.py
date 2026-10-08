import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        r = self.act(self.conv1(x))
        r = self.conv2(r)
        return x + 0.1 * r


class TinyAnimeSR(nn.Module):
    """
    Toy 2x super-resolution model:
    32x32 RGB -> 64x64 RGB

    Idea:
    1) bilinear upsample gives a blurry 64x64 baseline
    2) CNN predicts a residual correction
    3) output = baseline + residual
    """
    def __init__(self, channels=64, num_blocks=6):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        base = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        f = F.relu(self.head(base), inplace=True)
        f = self.body(f)
        residual = self.tail(f)
        return torch.clamp(base + residual, 0.0, 1.0)


# 输入低清图
# 3×32×32
#     │
#     ▼
# Bilinear ×2
#     │
#     ▼
# 3×64×64
#     │
#     ├──────────────────────────────┐
#     │                              │
#     ▼                              │
# Conv 3→64                          │
#     │                              │
# ReLU                               │
#     │                              │
# Residual Block ×6                  │
#     │                              │
# Conv 64→3                          │
#     │                              │
#     ▼                              │
# Residual                           │
# 3×64×64                            │
#     │                              │
#     └────────── + ◄────────────────┘
#                 │
#                 ▼
#             Clamp 0~1
#                 │
#                 ▼
#            高清预测图片
#             3×64×64