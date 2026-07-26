"""
红外 (Infrared / Thermal) 特征提取模块 — ResNet50 从头训练。

输入:  (B, 1, H, W)  红外热成像 (单通道灰度)
输出:  {"C2": (B,256,H/4,W/4),
        "C3": (B,512,H/8,W/8),
        "C4": (B,1024,H/16,W/16),
        "C5": (B,2048,H/32,W/32)}

策略: 第一层 conv1 输入通道=1，无法复用预训练权重，
      所有层 Kaiming 初始化，从头学习红外特有的热辐射特征。
"""

import torch
import torch.nn as nn
from typing import Dict
from net import ResNet50Backbone


class InfraredBranch(nn.Module):
    """红外分支 — ResNet50 从头训练。"""

    def __init__(self, frozen_stages: int = -1):
        super().__init__()
        self.backbone = ResNet50Backbone(
            in_channels=1,
            pretrained=False,         # 从头训练
            frozen_stages=frozen_stages,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.backbone(x)


# ── 快速验证 ──────────────────────────────────────────────────
def _test():
    print("=" * 50)
    print("[InfraredBranch] shape check")
    B, H, W = 2, 640, 640
    model = InfraredBranch()
    model.train()
    x = torch.randn(B, 1, H, W)
    feats = model(x)
    for k, v in feats.items():
        print(f"  {k}: {tuple(v.shape)}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
    print("  PASS")
    print("=" * 50)


if __name__ == "__main__":
    _test()
