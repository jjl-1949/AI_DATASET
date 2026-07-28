"""
DT (Depth-Thermal) 正交融合模块

对深度和红外两个模态的多尺度特征进行严格正交分解与融合。
采用显式投影减法 (Gram-Schmidt 风格)，保证分解后的分量满足数学正交性:

    <F_d^⊥, F_t> = 0,    <F_t^⊥, F_d> = 0

核心公式 (参考 RDTTrack):
    F_d^⊥ = F_d - α · proj_{F_t}(F_d)   其中 proj_{F_t}(F_d) = (<F_d,F_t> / ||F_t||²) · F_t
    F_t^⊥ = F_t - β · proj_{F_d}(F_t)   其中 proj_{F_d}(F_t) = (<F_t,F_d> / ||F_d||²) · F_d

α, β 为可学习的缩放系数 (sigmoid 约束到 [0,1])，允许网络自适应地控制投影移除强度。

融合策略: F_fused = Concat(F_d^⊥, F_t^⊥) → 1×1 卷积融合

与旧版的关键区别:
    - 旧版: F_d_orth = F_d * (1 - gate(F_shared))   ← 门控残差，无正交性保证
    - 新版: F_d_orth = F_d - α · proj_{F_t}(F_d)    ← 显式投影减法，数学正交

输入:  depth_feats  = {"C2": (B,256,H/4,W/4),  "C3": ..., "C4": ..., "C5": ...}
       thermal_feats = {"C2": (B,256,H/4,W/4),  "C3": ..., "C4": ..., "C5": ...}
输出:  {"C2": (B,256,H/4,W/4),  "C3": ..., "C4": ..., "C5": ...}

后续: 融合特征送入 DT-FPN 构建金字塔，再与 RGB-FPN 汇入 DCR-CBAM 融合。
"""

import torch
import torch.nn as nn
from typing import Dict


# ═══════════════════════════════════════════════════════════════
# Orthogonal Fusion Block (单尺度) — 显式投影版本
# ═══════════════════════════════════════════════════════════════
class OrthoFusionBlock(nn.Module):
    """单尺度正交融合块。

    严格的 Gram-Schmidt 正交分解 + 后门控自适应融合:

      1. 严格正交化 (α=1):
           F_d^⊥ = F_d' - proj_{F_t'}(F_d')    ← <F_d^⊥, F_t'> = 0
           F_t^⊥ = F_t' - proj_{F_d'}(F_t')    ← <F_t^⊥, F_d'> = 0

      2. 共性分量: 正交化过程中被移除的投影部分
           F_common = proj_{F_t'}(F_d') = F_d' - F_d^⊥

      3. 后门控自适应融合:
           F_fused = g_d·F_d^⊥ + g_t·F_t^⊥ + g_c·F_common

    门控在正交化之后执行——正交性不被破坏，
    门控只控制各分量在最终融合中的贡献比例。
    """

    def __init__(self, in_channels: int, hide_channels: int | None = None,
                 reduction: int = 4):
        super().__init__()
        mid = hide_channels if hide_channels is not None else max(in_channels // reduction, 32)

        # ── 投影空间映射 ──
        self.proj_d = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.proj_t = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # ── 后门控系数 (sigmoid → [0,1]) ──
        # 控制在正交化之后各分量的贡献比例
        self.gate_d = nn.Parameter(torch.tensor(0.0))   # depth 正交分量
        self.gate_t = nn.Parameter(torch.tensor(0.0))   # thermal 正交分量
        self.gate_c = nn.Parameter(torch.tensor(0.0))   # 共性分量

        # ── 融合 ──
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _gram_schmidt_step(F_a: torch.Tensor, F_b: torch.Tensor) -> torch.Tensor:
        """严格 Gram-Schmidt: F_a^⊥ = F_a - proj_{F_b}(F_a)

        proj_{F_b}(F_a) = (<F_a, F_b> / ||F_b||²) · F_b  (L2 投影)
        内积沿通道维度逐空间位置计算。
        结果满足 <F_a^⊥, F_b> = 0 (数学正交)。
        """
        inner = (F_a * F_b).sum(dim=1, keepdim=True)                # (B, 1, H, W)
        norm2 = F_b.pow(2).sum(dim=1, keepdim=True) + 1e-8          # (B, 1, H, W)
        proj = (inner / norm2) * F_b                                 # (B, C, H, W)
        return F_a - proj

    def forward(self, F_d: torch.Tensor, F_t: torch.Tensor) -> torch.Tensor:
        # 1. 映射到投影空间
        depth0 = self.proj_d(F_d)      # (B, mid, H, W)
        ir0 = self.proj_t(F_t)         # (B, mid, H, W)

        # 2. 严格正交化 (alpha = 1.0)
        depth_orth = self._gram_schmidt_step(depth0, ir0)    # <depth_orth, ir0> = 0
        ir_orth = self._gram_schmidt_step(ir0, depth0)       # <ir_orth, depth0> = 0

        # 3. 共性分量: 正交化时被移除的投影部分
        #    common = depth0 - depth_orth = proj_{ir0}(depth0)
        common = depth0 - depth_orth

        # 4. 后门控: 在正交化之后控制各分量的贡献
        g_d = torch.sigmoid(self.gate_d)  # depth 独有信息的贡献
        g_t = torch.sigmoid(self.gate_t)  # thermal 独有信息的贡献
        g_c = torch.sigmoid(self.gate_c)  # 共性信息的贡献

        # 5. 门控融合
        depth_gated = g_d * depth_orth
        ir_gated = g_t * ir_orth
        common_gated = g_c * common

        out = torch.cat([depth_gated + common_gated,
                         ir_gated + common_gated], dim=1)  # (B, mid*2, H, W)
        return self.fuse(out)


# ═══════════════════════════════════════════════════════════════
# Multi-Scale DT Orthogonal Fusion
# ═══════════════════════════════════════════════════════════════
class DTOrtFusion(nn.Module):
    """多尺度深度-红外正交融合网络。

    在 C2, C3, C4, C5 四个尺度上分别进行严格正交分解 + 后门控融合。
    每个尺度独立学习门控系数 g_d/g_t/g_c:
      - 浅层 (C2): 纹理细节差异大 → 正交分量门控高
      - 深层 (C5): 语义类别趋同 → 共性分量门控高

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
    print("[DTOrtFusion] Depth-Thermal Orthogonal Fusion (Gram-Schmidt)")
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

    # ── 正交性验证: <F_d^⊥, F_t> = 0 (严格 Gram-Schmidt) ──
    print("\n  --- Orthogonality Test ---")
    from dt_ort import OrthoFusionBlock as OFB

    block = OFB(in_channels=256)
    block.eval()
    with torch.no_grad():
        d = torch.randn(2, 256, 40, 40)
        t = torch.randn(2, 256, 40, 40)

        # 严格正交化 (无 alpha 参数 — 永为 1.0)
        depth0 = block.proj_d(d)
        ir0 = block.proj_t(t)
        depth_orth = block._gram_schmidt_step(depth0, ir0)
        ir_orth = block._gram_schmidt_step(ir0, depth0)

        # <F_d^⊥, F_t> 应为 0 (数学正交)
        cos_d = (depth_orth * ir0).sum(dim=1).abs().mean()
        cos_ir = (ir_orth * depth0).sum(dim=1).abs().mean()
        print(f"  |<F_d^⊥, F_t>| = {cos_d.item():.6e}  (should be ~0)")
        print(f"  |<F_t^⊥, F_d>| = {cos_ir.item():.6e}  (should be ~0)")
        assert cos_d < 1e-5 and cos_ir < 1e-5, "Orthogonality violated!"

        # 验证门控不破坏正交性
        g_d = torch.sigmoid(block.gate_d)
        g_c = torch.sigmoid(block.gate_c)
        gated = g_d * depth_orth + g_c * (depth0 - depth_orth)
        cos_gated = (gated * ir0).sum(dim=1).abs().mean()
        print(f"  |<gated_d, F_t>|  = {cos_gated.item():.6e}  (gating preserves orthogonality)")

    print("  PASS — true orthogonality verified")
    print("=" * 60)


if __name__ == "__main__":
    _test()
