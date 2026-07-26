"""
DT (Depth-Thermal) 正交融合模块

对深度和红外两个模态的多尺度特征进行正交分解与融合。
核心思想：将每个模态的特征分解为"共享分量"和"正交分量"——
- 共享分量: 两个模态共有的语义信息（如物体边界、语义类别一致性）
- 正交分量: 每个模态独有的互补信息（深度→几何结构，红外→热辐射特征）

融合策略: F_fused = F_shared + F_d^⊥ + F_t^⊥
既保留了共性语义，又保留了各传感器独有的判别力。

输入:  depth_feats  = {"C2": (B,256,H/4,W/4),  "C3": ..., "C4": ..., "C5": ...}
       thermal_feats = {"C2": (B,256,H/4,W/4),  "C3": ..., "C4": ..., "C5": ...}
输出:  {"C2": (B,256,H/4,W/4),  "C3": ..., "C4": ..., "C5": ...}

后续: 融合特征送入 DT-FPN 构建金字塔，再与 RGB-FPN 汇入 DCR-CBAM 融合。
"""

import torch
import torch.nn as nn
from typing import Dict


# ═══════════════════════════════════════════════════════════════
# Orthogonal Fusion Block (单尺度)
# ═══════════════════════════════════════════════════════════════
class OrthoFusionBlock(nn.Module):
    """单尺度正交融合块。

    对给定尺度的深度特征 F_d 和红外特征 F_t:
      1. 拼接后提取共享语义 F_shared
      2. 用可学习的门控抑制各模态中被共享分量解释的部分
      3. 残差即为各模态独有的正交分量
      4. 融合三者：共享 + 深度正交 + 红外正交
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(in_channels // reduction, 32)

        # ── 共享语义提取 ──
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # ── 门控: 学习共享特征在各模态空间中的投影强度 ──
        self.gate_d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )
        self.gate_t = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

        # ── 融合后精炼 ──
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, F_d: torch.Tensor, F_t: torch.Tensor) -> torch.Tensor:
        # 1. 共享语义: 从拼接特征中聚合共性信息
        cat = torch.cat([F_d, F_t], dim=1)
        F_shared = self.shared_conv(cat)             # (B, C, H, W)

        # 2. 正交分量: gate → 1 表示"被共享分量解释"，残差为独有信息
        F_d_orth = F_d * (1.0 - self.gate_d(F_shared))
        F_t_orth = F_t * (1.0 - self.gate_t(F_shared))

        # 3. 互补融合: 共性 + 深度独有 + 红外独有
        out = F_shared + F_d_orth + F_t_orth
        return self.fuse(out)


# ═══════════════════════════════════════════════════════════════
# Multi-Scale DT Orthogonal Fusion
# ═══════════════════════════════════════════════════════════════
class DTOrtFusion(nn.Module):
    """多尺度深度-红外正交融合网络。

    在 C2, C3, C4, C5 四个尺度上分别进行正交融合。
    每个尺度独立学习门控参数，因为不同尺度的共享/正交比例不同：
      - 浅层 (C2): 纹理细节 → 正交分量占比高
      - 深层 (C5): 语义类别 → 共享分量占比高

    通道配置 (ResNet50 骨干):
        C2: 256,  C3: 512,  C4: 1024,  C5: 2048
    """

    CHANNELS: Dict[str, int] = {"C2": 256, "C3": 512, "C4": 1024, "C5": 2048}

    def __init__(self, channels: Dict[str, int] | None = None):
        super().__init__()
        ch = channels if channels is not None else self.CHANNELS

        self.fusion_blocks = nn.ModuleDict({
            level: OrthoFusionBlock(ch[level])
            for level in ch
        })

    def forward(self,
                depth_feats: Dict[str, torch.Tensor],
                thermal_feats: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """前向传播。

        Args:
            depth_feats:  深度分支多尺度特征 {"C2":..., "C3":..., "C4":..., "C5":...}
            thermal_feats: 红外分支多尺度特征 (同上结构)

        Returns:
            正交融合后的多尺度特征字典
        """
        fused: Dict[str, torch.Tensor] = {}
        for level, block in self.fusion_blocks.items():
            fused[level] = block(depth_feats[level], thermal_feats[level])
        return fused


# ── 快速验证 ──────────────────────────────────────────────────
def _test():
    print("=" * 60)
    print("[DTOrtFusion] Depth-Thermal Orthogonal Fusion")
    B, H, W = 2, 640, 640

    model = DTOrtFusion()
    model.train()

    # 模拟深度 & 红外分支输出 (ResNet50 多尺度特征)
    depth_feats = {
        "C2": torch.randn(B, 256,  H // 4,  W // 4),
        "C3": torch.randn(B, 512,  H // 8,  W // 8),
        "C4": torch.randn(B, 1024, H // 16, W // 16),
        "C5": torch.randn(B, 2048, H // 32, W // 32),
    }
    thermal_feats = {k: torch.randn_like(v) for k, v in depth_feats.items()}

    fused = model(depth_feats, thermal_feats)

    for k, v in fused.items():
        # 输出维度应与输入一致
        expected = depth_feats[k].shape
        assert v.shape == expected, f"{k}: {tuple(v.shape)} != {tuple(expected)}"
        print(f"  {k}: {tuple(v.shape)}  OK")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
    print("  PASS")
    print("=" * 60)


if __name__ == "__main__":
    _test()
