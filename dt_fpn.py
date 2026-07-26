"""
DT FPN (Depth-Thermal Feature Pyramid Network) — 深度-红外融合特征金字塔

在 DT 正交融合 (DTOrtFusion) 输出的多尺度特征之上构建特征金字塔。
DT 正交融合保留了各尺度的通道数不变 (C2:256, C3:512, C4:1024, C5:2048)，
因此 FPN 结构与 RGB 分支完全一致。

输入:  {"C2": (B,256, H/4, W/4),
        "C3": (B,512, H/8, W/8),
        "C4": (B,1024,H/16,W/16),
        "C5": (B,2048,H/32,W/32)}     ← 来自 DTOrtFusion 的输出

输出:  {"P2": (B,256,H/4,W/4),
        "P3": (B,256,H/8,W/8),
        "P4": (B,256,H/16,W/16),
        "P5": (B,256,H/32,W/32)}

数据流:  Depth ResNet50  ─┐
                          ├──→ DTOrtFusion ──→ DTFPN ──→ DCR-CBAM ←──
        Thermal ResNet50 ─┘                                        │
        RGB ResNet50 ────→ RGBFPN ─────────────────────────────────┘

后续: DT-FPN 输出与 RGB-FPN 输出在 DCR-CBAM 中进行跨模态注意力融合。
"""

import torch
import torch.nn as nn
from typing import Dict

from rgb_fpn import FPN


# ═══════════════════════════════════════════════════════════════
# DT 专用 FPN
# ═══════════════════════════════════════════════════════════════
class DTFPN(nn.Module):
    """深度-红外融合特征金字塔。

    接收 DTOrtFusion 的正交融合多尺度特征，
    通过 FPN 生成统一 256 通道的多尺度金字塔特征。

    输入通道与 ResNet50 骨干一致:
        C2: 256,  C3: 512,  C4: 1024,  C5: 2048

    输出金字塔全部为 256 通道:
        P2: H/4,   P3: H/8,   P4: H/16,  P5: H/32
    """

    # DTOrtFusion 输出的各级通道数 (与 ResNet50 一致)
    IN_CHANNELS: Dict[str, int] = {"C2": 256, "C3": 512, "C4": 1024, "C5": 2048}

    def __init__(self, out_channels: int = 256, extra_level: bool = False):
        super().__init__()
        self.fpn = FPN(
            in_channels=self.IN_CHANNELS,
            out_channels=out_channels,
            extra_level=extra_level,
        )

    def forward(self, feats: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """前向传播。

        Args:
            feats: DTOrtFusion 输出 {"C2":..., "C3":..., "C4":..., "C5":...}

        Returns:
            金字塔特征 {"P2":..., "P3":..., "P4":..., "P5":...}
        """
        return self.fpn(feats)


# ── 快速验证 ──────────────────────────────────────────────────
def _test():
    print("=" * 60)
    print("[DTFPN] Depth-Thermal Feature Pyramid Network")

    B, H, W = 2, 640, 640

    # 模拟 DTOrtFusion 输出 (通道数 = ResNet50 各级通道数)
    dt_feats = {
        "C2": torch.randn(B, 256,  H // 4,  W // 4),
        "C3": torch.randn(B, 512,  H // 8,  W // 8),
        "C4": torch.randn(B, 1024, H // 16, W // 16),
        "C5": torch.randn(B, 2048, H // 32, W // 32),
    }

    model = DTFPN(out_channels=256)
    model.train()

    out = model(dt_feats)

    expected = {"P2": (B, 256, H//4, W//4),
                "P3": (B, 256, H//8, W//8),
                "P4": (B, 256, H//16, W//16),
                "P5": (B, 256, H//32, W//32)}

    for k, v in out.items():
        exp = expected[k]
        assert v.shape == exp, f"{k}: {tuple(v.shape)} != {exp}"
        print(f"  {k}: {tuple(v.shape)}  OK")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
    print("  PASS")
    print("=" * 60)


if __name__ == "__main__":
    _test()
