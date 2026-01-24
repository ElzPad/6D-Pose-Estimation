import torch
import torch.nn as nn

class DepthFeatureExtractor(nn.Module):
    """Simple CNN that turns a (masked) depth crop into a compact feature vector.

    Expected input: (B, 1, H, W) depth_mask in [0, 1] (zeros = invalid/background).
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim

        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            block(1, 32, 2),   # 224 -> 112
            block(32, 64, 2),  # 112 -> 56
            block(64, 128, 2), # 56 -> 28
            block(128, 256, 2),# 28 -> 14
            nn.Conv2d(256, out_dim, kernel_size=3, stride=2, padding=1, bias=False), # 14 -> 7
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, depth_mask: torch.Tensor) -> torch.Tensor:
        if depth_mask.dim() != 4:
            raise ValueError(f"depth_mask must be (B,1,H,W), got {tuple(depth_mask.shape)}")
        if depth_mask.size(1) != 1:
            # allow (B,3,H,W) by converting to grayscale-ish
            depth_mask = depth_mask[:, :1, ...]
        x = self.net(depth_mask)
        return x.flatten(1)  # (B, out_dim)
