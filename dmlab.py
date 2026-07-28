"""
DMLab (DeepLabV3+ Decoder) — 多尺度上下文解码器

在 DCR-CBAM 融合的多尺度特征之上，通过 ASPP 捕获多尺度上下文，
逐级上采样并与各层 skip connection 融合，输出高分辨率特征图。

结构:
    P5 ──→ ASPP → upsample×2 ──[+P4 skip]──→ upsample×2 ──[+P3 skip]──→ upsample×2 ──[+P2 skip]──→ output

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
        # NOTE: BN omitted here because AdaptiveAvgPool2d(1) produces (B,C,1,1)
        # which causes BN to fail with batch_size=1 in training mode.
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
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
# DMLab Decoder (DeepLabV3+ style, multi-level skip)
# ═══════════════════════════════════════════════════════════════
class DMLabDecoder(nn.Module):
    """DeepLabV3+ 风格解码器 (多级 skip)。

    P5: ASPP 捕获多尺度上下文
    P4, P3, P2: 逐级上采样 + skip connection
    全部分支参与前向，确保所有 DCR-CBAM 层级都能获得梯度。

    结构:
        P5 ──→ ASPP ──→ upsample×2 ─┬─→ conv ──→ upsample×2 ─┬─→ conv ──→ upsample×2 ─┬─→ conv → output
                       P4 ──[1×1]──┘            P3 ──[1×1]──┘            P2 ──[1×1]──┘
    """

    def __init__(self,
                 channels: int = 256,
                 skip_out: int = 48,
                 out_channels: int = 256,
                 aspp_rates: list[int] | None = None):
        super().__init__()

        # ── ASPP on P5 (coarsest) ──
        self.aspp = ASPP(channels, channels, aspp_rates)

        # ── Skip projections: P4, P3, P2 → skip_out channels ──
        # P4 skip (stride 16)
        self.skip_p4 = nn.Sequential(
            nn.Conv2d(channels, skip_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_out),
            nn.ReLU(inplace=True),
        )
        # P3 skip (stride 8)
        self.skip_p3 = nn.Sequential(
            nn.Conv2d(channels, skip_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_out),
            nn.ReLU(inplace=True),
        )
        # P2 skip (stride 4)
        self.skip_p2 = nn.Sequential(
            nn.Conv2d(channels, skip_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_out),
            nn.ReLU(inplace=True),
        )

        # ── Fusion after each upsample step ──
        # P5→P4: ASPP out (channels) + P4 skip (skip_out)
        self.fuse_p4 = nn.Sequential(
            nn.Conv2d(channels + skip_out, channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # P4→P3: fused (channels) + P3 skip (skip_out)
        self.fuse_p3 = nn.Sequential(
            nn.Conv2d(channels + skip_out, channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # P3→P2: fused (channels) + P2 skip (skip_out)
        self.fuse_p2 = nn.Sequential(
            nn.Conv2d(channels + skip_out, channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # ── Final refinement ──
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=3,
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

    def forward(self, p2: torch.Tensor, p3: torch.Tensor,
                p4: torch.Tensor, p5: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            p2: stride-4  特征 (B, channels, H/4,  W/4)
            p3: stride-8  特征 (B, channels, H/8,  W/8)
            p4: stride-16 特征 (B, channels, H/16, W/16)
            p5: stride-32 特征 (B, channels, H/32, W/32)

        Returns:
            解码后特征 (B, out_channels, H/4, W/4)
        """
        # 1. ASPP on P5
        feat = self.aspp(p5)                            # (B, 256, H/32, W/32)

        # 2. P5→P4: upsample ×2, concat with P4 skip
        feat = F.interpolate(feat, size=p4.shape[2:],
                             mode="bilinear", align_corners=False)
        skip4 = self.skip_p4(p4)
        feat = self.fuse_p4(torch.cat([feat, skip4], dim=1))  # (B, 256, H/16, W/16)

        # 3. P4→P3: upsample ×2, concat with P3 skip
        feat = F.interpolate(feat, size=p3.shape[2:],
                             mode="bilinear", align_corners=False)
        skip3 = self.skip_p3(p3)
        feat = self.fuse_p3(torch.cat([feat, skip3], dim=1))  # (B, 256, H/8, W/8)

        # 4. P3→P2: upsample ×2, concat with P2 skip
        feat = F.interpolate(feat, size=p2.shape[2:],
                             mode="bilinear", align_corners=False)
        skip2 = self.skip_p2(p2)
        feat = self.fuse_p2(torch.cat([feat, skip2], dim=1))  # (B, 256, H/4, W/4)

        # 5. Final refinement
        return self.output_conv(feat)


# ═══════════════════════════════════════════════════════════════
# DMLab 顶层封装
# ═══════════════════════════════════════════════════════════════
class DMLab(nn.Module):
    """DMLab 多尺度上下文解码器。

    接收 DCR-CBAM 融合后的多尺度特征 (P2–P5)，
    通过 ASPP 多尺度上下文捕获 + 逐级 skip connection，
    输出单一 stride-4 高分辨率特征图。

    Args:
        channels:       特征通道数 (默认 256)
        skip_out:       skip 分支输出通道数 (默认 48)
        out_channels:   最终输出通道数 (默认 256)
    """

    def __init__(self,
                 channels: int = 256,
                 skip_out: int = 48,
                 out_channels: int = 256):
        super().__init__()
        self.decoder = DMLabDecoder(
            channels=channels,
            skip_out=skip_out,
            out_channels=out_channels,
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播。

        Args:
            feats: DCR-CBAM 融合特征 {"P2":..., "P3":..., "P4":..., "P5":...}

        Returns:
            解码后特征 (B, out_channels, H/4, W/4)
        """
        return self.decoder(feats["P2"], feats["P3"], feats["P4"], feats["P5"])


# ── 快速验证 ──────────────────────────────────────────────────
def _test():
    print("=" * 60)
    print("[DMLab] DeepLabV3+ Decoder (multi-level skip)")
    B, H, W = 2, 640, 640

    # 模拟 DCR-CBAM 融合后的多尺度特征
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
    assert aspp_out.shape == feats["P5"].shape
    print(f"  Input:  {tuple(feats['P5'].shape)}")
    print(f"  Output: {tuple(aspp_out.shape)}  OK")

    # ── 测试 DMLabDecoder (新: 多级 skip) ──
    print("\n[DMLabDecoder] P5→P4→P3→P2 decoder")
    decoder = DMLabDecoder(channels=256, skip_out=48, out_channels=256)
    decoder.train()
    dec_out = decoder(feats["P2"], feats["P3"], feats["P4"], feats["P5"])
    expected_shape = (B, 256, H // 4, W // 4)
    assert dec_out.shape == expected_shape, \
        f"Decoder: {tuple(dec_out.shape)} != {expected_shape}"
    print(f"  P5 (input):  {tuple(feats['P5'].shape)}")
    print(f"  P4 (skip):   {tuple(feats['P4'].shape)}")
    print(f"  P3 (skip):   {tuple(feats['P3'].shape)}")
    print(f"  P2 (skip):   {tuple(feats['P2'].shape)}")
    print(f"  Output:      {tuple(dec_out.shape)}  OK")

    # ── 测试 DMLab 顶层 ──
    print("\n[DMLab]")
    model = DMLab(channels=256, skip_out=48, out_channels=256)
    model.train()
    out = model(feats)
    assert out.shape == (B, 256, H // 4, W // 4)
    print(f"  Input keys: {list(feats.keys())}")
    print(f"  Output:     {tuple(out.shape)}  OK")

    # ── 非正方形输入测试 ──
    print("\n[Non-square input 360x640]")
    H2, W2 = 360, 640
    feats2 = {
        "P2": torch.randn(B, 256, H2 // 4,  W2 // 4),
        "P3": torch.randn(B, 256, H2 // 8,  W2 // 8),
        "P4": torch.randn(B, 256, H2 // 16, W2 // 16),
        "P5": torch.randn(B, 256, H2 // 32, W2 // 32),
    }
    dec_out2 = decoder(feats2["P2"], feats2["P3"], feats2["P4"], feats2["P5"])
    assert dec_out2.shape == feats2["P2"].shape
    print(f"  P2:     {tuple(feats2['P2'].shape)}")
    print(f"  Output: {tuple(dec_out2.shape)}  OK")

    # ── 梯度流验证: 所有 skip 层级必须获得梯度 ──
    print("\n[Gradient flow: all levels]")
    model.train()
    dmlab_out = model(feats)
    loss = dmlab_out.mean()
    loss.backward()
    grad_ok = all(
        p.grad is not None
        for n, p in model.named_parameters()
        if p.requires_grad
    )
    if grad_ok:
        print(f"  ALL parameters have gradients: OK")
    else:
        missing = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
        print(f"  MISSING: {missing}")

    # ── 参数量 ──
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  params: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
    print("  PASS")
    print("=" * 60)


if __name__ == "__main__":
    _test()
