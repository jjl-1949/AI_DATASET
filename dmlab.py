"""
DMLab (DeepLabV3+ Decoder) — 多尺度上下文解码器

在 DCR-CBAM 融合的多尺度特征之上，通过 ASPP 捕获多尺度上下文，
结合 P2 低层特征恢复空间细节，输出高分辨率特征图供检测头使用。

结构:
    P5 ──→ ASPP (空洞金字塔池化) → high_level (256ch, /32)
                                          │ upsample ×8
    P2 ──→ 1×1 conv → low_level (48ch) ──[concat]──→ 3×3 convs → output (256ch, /4)

ASPP 分支:
    1×1 conv | 3×3 rate=6 | 3×3 rate=12 | 3×3 rate=18 | Global AvgPool
    所有分支 concat → 1×1 projection → 256ch

输入:  {"P2": (B,256,H/4,W/4),
        "P3": (B,256,H/8,W/8),
        "P4": (B,256,H/16,W/16),
        "P5": (B,256,H/32,W/32)}     ← 来自 DCR-CBAM 的输出

输出:  (B, 256, H/4, W/4)              ← 高分辨率特征图

后续: DMLab 输出送入 Detection Head (det_head.py) 进行目标检测。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════
# ASPP (Atrous Spatial Pyramid Pooling)
# ═══════════════════════════════════════════════════════════════
class ASPP(nn.Module):
    """空洞空间金字塔池化。

    在多个空洞率下并行提取上下文，覆盖不同尺度的感受野:
      - rate=6:  感受野 ~13×13 → 适合中等目标
      - rate=12: 感受野 ~25×25 → 适合大目标
      - rate=18: 感受野 ~37×37 → 适合超大目标
      - Global pooling: 全局上下文

    Args:
        in_channels:  输入通道数 (默认 256)
        out_channels: 输出通道数 (默认 256)
        rates:        空洞率列表 (默认 [6, 12, 18])
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256,
                 rates: list[int] | None = None):
        super().__init__()
        if rates is None:
            rates = [6, 12, 18]

        # ── 1×1 卷积分支 ──
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # ── 空洞卷积分支 ──
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for r in rates
        ])

        # ── 全局平均池化分支 ──
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # ── 融合投影: concat 5 分支 → out_channels ──
        num_branches = 1 + len(rates) + 1  # 1×1 + dilated + global
        self.proj = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]

        # 1×1
        feat_1x1 = self.conv_1x1(x)

        # Dilated convs
        dilated_feats = [conv(x) for conv in self.dilated_convs]

        # Global pooling → upsample
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w),
                                    mode="bilinear", align_corners=False)

        # Concat all branches
        concat = torch.cat([feat_1x1] + dilated_feats + [global_feat], dim=1)

        return self.proj(concat)


# ═══════════════════════════════════════════════════════════════
# DMLab Decoder (DeepLabV3+ style)
# ═══════════════════════════════════════════════════════════════
class DMLabDecoder(nn.Module):
    """DeepLabV3+ 风格解码器。

    高层特征 (P5): ASPP 捕获多尺度上下文 → 上采样 ×8
    低层特征 (P2): 1×1 卷积降维 → skip connection
    拼接后 3×3 卷积精炼 → 输出 stride-4 高分辨率特征图
    """

    def __init__(self,
                 low_level_channels: int = 256,
                 high_level_channels: int = 256,
                 low_level_out: int = 48,
                 out_channels: int = 256,
                 aspp_rates: list[int] | None = None):
        super().__init__()

        # ── ASPP on high-level (P5) ──
        self.aspp = ASPP(high_level_channels, high_level_channels, aspp_rates)

        # ── Low-level projection (P2) ──
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_out, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(low_level_out),
            nn.ReLU(inplace=True),
        )

        # ── High-level after upsample ──
        self.high_level_conv = nn.Sequential(
            nn.Conv2d(high_level_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # ── Fusion: low_level_out + out_channels → out_channels ──
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(low_level_out + out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, low_level: torch.Tensor,
                high_level: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            low_level:  P2 特征 (B, low_level_channels, H/4, W/4)
            high_level: P5 特征 (B, high_level_channels, H/32, W/32)

        Returns:
            解码后特征 (B, out_channels, H/4, W/4)
        """
        # 1. ASPP on high-level
        high = self.aspp(high_level)          # (B, 256, H/32, W/32)

        # 2. Upsample high-level ×8 to match low-level spatial size
        high_up = F.interpolate(high, size=low_level.shape[2:],
                                mode="bilinear", align_corners=False)
        high_up = self.high_level_conv(high_up)  # (B, 256, H/4, W/4)

        # 3. Low-level projection
        low = self.low_level_conv(low_level)     # (B, 48, H/4, W/4)

        # 4. Concat + refine
        fused = torch.cat([high_up, low], dim=1)  # (B, 304, H/4, W/4)
        fused = self.fuse_conv1(fused)            # (B, 256, H/4, W/4)
        fused = self.fuse_conv2(fused)            # (B, 256, H/4, W/4)

        return fused


# ═══════════════════════════════════════════════════════════════
# DMLab 顶层封装
# ═══════════════════════════════════════════════════════════════
class DMLab(nn.Module):
    """DMLab 多尺度上下文解码器。

    接收 DCR-CBAM 融合后的多尺度特征 (P2–P5)，
    通过 ASPP 多尺度上下文捕获 + P2 skip connection 空间细节恢复，
    输出单一 stride-4 高分辨率特征图。

    Args:
        low_level_key:  低层特征的 key (默认 "P2")
        high_level_key: 高层特征的 key (默认 "P5")
        low_level_channels: 低层输入通道 (默认 256)
        high_level_channels: 高层输入通道 (默认 256)
        out_channels:   输出通道数 (默认 256)
    """

    def __init__(self,
                 low_level_key: str = "P2",
                 high_level_key: str = "P5",
                 low_level_channels: int = 256,
                 high_level_channels: int = 256,
                 out_channels: int = 256):
        super().__init__()
        self.low_key = low_level_key
        self.high_key = high_level_key

        self.decoder = DMLabDecoder(
            low_level_channels=low_level_channels,
            high_level_channels=high_level_channels,
            out_channels=out_channels,
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播。

        Args:
            feats: DCR-CBAM 融合特征 {"P2":..., "P3":..., "P4":..., "P5":...}

        Returns:
            解码后特征 (B, out_channels, H/4, W/4)
        """
        return self.decoder(feats[self.low_key], feats[self.high_key])


# ── 快速验证 ──────────────────────────────────────────────────
def _test():
    print("=" * 60)
    print("[DMLab] DeepLabV3+ Decoder")
    B, H, W = 2, 640, 640

    # 模拟 DCR-CBAM 融合后的多尺度特征 (FPN 输出格式)
    feats = {
        "P2": torch.randn(B, 256, H // 4,  W // 4),
        "P3": torch.randn(B, 256, H // 8,  W // 8),
        "P4": torch.randn(B, 256, H // 16, W // 16),
        "P5": torch.randn(B, 256, H // 32, W // 32),
    }

    # ── 测试 ASPP ──
    print("\n[ASPP]")
    aspp = ASPP(in_channels=256, out_channels=256)
    aspp.train()
    aspp_out = aspp(feats["P5"])
    assert aspp_out.shape == feats["P5"].shape, \
        f"ASPP: {tuple(aspp_out.shape)} != {tuple(feats['P5'].shape)}"
    print(f"  Input:  {tuple(feats['P5'].shape)}")
    print(f"  Output: {tuple(aspp_out.shape)}  OK")

    # ── 测试 DMLabDecoder ──
    print("\n[DMLabDecoder]")
    decoder = DMLabDecoder(
        low_level_channels=256, high_level_channels=256,
        low_level_out=48, out_channels=256,
    )
    decoder.train()
    dec_out = decoder(feats["P2"], feats["P5"])
    expected_shape = (B, 256, H // 4, W // 4)
    assert dec_out.shape == expected_shape, \
        f"Decoder: {tuple(dec_out.shape)} != {expected_shape}"
    print(f"  P2 (low):   {tuple(feats['P2'].shape)}")
    print(f"  P5 (high):  {tuple(feats['P5'].shape)}")
    print(f"  Output:     {tuple(dec_out.shape)}  OK")

    # ── 测试 DMLab 顶层 ──
    print("\n[DMLab]")
    model = DMLab()
    model.train()
    out = model(feats)
    assert out.shape == (B, 256, H // 4, W // 4), \
        f"DMLab: {tuple(out.shape)} != (B, 256, H//4, W//4)"
    print(f"  Input keys: {list(feats.keys())}")
    print(f"  Output:     {tuple(out.shape)}  OK")

    # ── 非正方形输入测试 ──
    print("\n[Non-square input]")
    H2, W2 = 360, 640
    feats2 = {
        "P2": torch.randn(B, 256, H2 // 4,  W2 // 4),
        "P5": torch.randn(B, 256, H2 // 32, W2 // 32),
    }
    dec_out2 = decoder(feats2["P2"], feats2["P5"])
    assert dec_out2.shape == feats2["P2"].shape, \
        f"Non-square: {tuple(dec_out2.shape)} != {tuple(feats2['P2'].shape)}"
    print(f"  Input:  {tuple(feats2['P2'].shape)} (stride 4)")
    print(f"  Output: {tuple(dec_out2.shape)}  OK")

    # ── 参数量 ──
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  params: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
    print("  PASS")
    print("=" * 60)


if __name__ == "__main__":
    _test()
