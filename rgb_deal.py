"""
RGB (可见光) 特征提取模块 — ResNet50 迁移学习。

输入:  (B, 3, H, W)  RGB 图像
输出:  {"C2": (B,256,H/4,W/4),
        "C3": (B,512,H/8,W/8),
        "C4": (B,1024,H/16,W/16),
        "C5": (B,2048,H/32,W/32)}

策略: ImageNet 预训练权重，默认冻结 stem + layer1，微调高层。
"""

import torch
import torch.nn as nn
from typing import Dict
from net import ResNet50Backbone


class RGBBranch(nn.Module):
    """RGB 分支 — ResNet50 迁移学习。"""

    def __init__(self, frozen_stages: int = 1):
        super().__init__()
        self.backbone = ResNet50Backbone(
            in_channels=3,
            pretrained=True,          # ImageNet 预训练
            frozen_stages=frozen_stages,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.backbone(x)


# ── 快速验证 ──────────────────────────────────────────────────
def _test():
    print("=" * 50)
    print("[RGBBranch] shape check")
    B, H, W = 2, 640, 640
    model = RGBBranch(frozen_stages=1)
    model.train()
    x = torch.randn(B, 3, H, W)
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
