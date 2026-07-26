"""
RGB FPN (Feature Pyramid Network) — 可见光特征金字塔

在 ResNet50 骨干输出的多尺度特征 (C2–C5) 之上构建自顶向下的特征金字塔。
通过横向连接和上采样融合，在多个尺度上生成语义强、分辨率高的特征图，
用于后续的多尺度目标检测。

输入:  {"C2": (B,256, H/4, W/4),
        "C3": (B,512, H/8, W/8),
        "C4": (B,1024,H/16,W/16),
        "C5": (B,2048,H/32,W/32)}

输出:  {"P2": (B,256,H/4,W/4),
        "P3": (B,256,H/8,W/8),
        "P4": (B,256,H/16,W/16),
        "P5": (B,256,H/32,W/32)}

结构:  C5 ──[1×1]──→ M5 ──[3×3]──→ P5
                        │ (↑×2)
       C4 ──[1×1]──→ + ──[3×3]──→ P4
                        │ (↑×2)
       C3 ──[1×1]──→ + ──[3×3]──→ P3
                        │ (↑×2)
       C2 ──[1×1]──→ + ──[3×3]──→ P2

后续: RGB-FPN 输出与 DT-FPN 输出汇入 DCR-CBAM 进行跨模态特征融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════
# FPN 核心
# ═══════════════════════════════════════════════════════════════
class FPN(nn.Module):
    """特征金字塔网络 (Feature Pyramid Network)。

    标准 FPN 实现:
      - 1×1 横向卷积将各级骨干特征统一到 out_channels
      - 自顶向下逐层上采样并相加
      - 3×3 卷积消除上采样混叠伪影

    Args:
        in_channels:    各级输入通道映射 {"C2": 256, "C3": 512, "C4": 1024, "C5": 2048}
        out_channels:   输出通道数 (默认 256)
        extra_level:    是否额外生成 P6 (stride=64, 在 P5 上 stride-2 池化)
    """

    def __init__(self,
                 in_channels: Dict[str, int],
                 out_channels: int = 256,
                 extra_level: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.extra_level = extra_level

        # ── 按尺度排序 (C2→C5, 深→浅) ──
        self.levels = sorted(in_channels.keys())  # ["C2","C3","C4","C5"]

        # ── 横向连接: 1×1 卷积统一通道数 (按名称索引) ──
        self.lateral_convs = nn.ModuleDict({
            lv: nn.Conv2d(in_channels[lv], out_channels, kernel_size=1, bias=False)
            for lv in self.levels
        })

        # ── 输出卷积: 3×3 消除上采样混叠 ──
        self.output_convs = nn.ModuleDict({
            lv: nn.Conv2d(out_channels, out_channels, kernel_size=3,
                          padding=1, bias=False)
            for lv in self.levels
        })

        # ── P6 层 (可选) ──
        if extra_level:
            self.p6_pool = nn.MaxPool2d(kernel_size=1, stride=2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feats: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """前向传播。

        Args:
            feats: 骨干多尺度特征 {"C2": ..., "C3": ..., "C4": ..., "C5": ...}

        Returns:
            金字塔特征 {"P2": ..., "P3": ..., "P4": ..., "P5": ...}
        """
        # ── 先计算所有横向连接 ──
        laterals = {lv: self.lateral_convs[lv](feats[lv]) for lv in self.levels}

        # ── 自顶向下通路 (C5 → C4 → C3 → C2) ──
        reversed_levels = list(reversed(self.levels))  # ["C5","C4","C3","C2"]

        pyramid: Dict[str, torch.Tensor] = {}
        prev = laterals[reversed_levels[0]]              # M5
        pyramid[reversed_levels[0]] = self.output_convs[reversed_levels[0]](prev)  # P5

        for lv in reversed_levels[1:]:
            up = F.interpolate(prev, size=laterals[lv].shape[2:],
                               mode="nearest")
            prev = laterals[lv] + up
            pyramid[lv] = self.output_convs[lv](prev)

        # ── 重命名 C→P ──
        out = {k.replace("C", "P"): v for k, v in pyramid.items()}

        # ── 可选 P6 ──
        if self.extra_level:
            out["P6"] = self.p6_pool(out["P5"])

        return out


# ═══════════════════════════════════════════════════════════════
# RGB 专用 FPN
# ═══════════════════════════════════════════════════════════════
class RGBFPN(nn.Module):
    """RGB 分支特征金字塔。

    接收 ResNet50 RGB 骨干的多尺度输出 (C2–C5)，
    通过 FPN 生成统一 256 通道的多尺度金字塔特征 (P2–P5)。
    """

    # ResNet50 各级通道数
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
            feats: RGB 骨干输出 {"C2":..., "C3":..., "C4":..., "C5":...}

        Returns:
            金字塔特征 {"P2":..., "P3":..., "P4":..., "P5":...}
        """
        return self.fpn(feats)


# ── 快速验证 ──────────────────────────────────────────────────
def _test():
    print("=" * 60)
    print("[RGBFPN] Feature Pyramid Network")

    B, H, W = 2, 640, 640

    # 模拟 RGB ResNet50 骨干输出
    in_feats = {
        "C2": torch.randn(B, 256,  H // 4,  W // 4),
        "C3": torch.randn(B, 512,  H // 8,  W // 8),
        "C4": torch.randn(B, 1024, H // 16, W // 16),
        "C5": torch.randn(B, 2048, H // 32, W // 32),
    }

    model = RGBFPN(out_channels=256)
    model.train()

    out = model(in_feats)

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
