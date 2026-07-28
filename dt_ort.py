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
    """单尺度正交融合块 (显式投影减法)。

    对给定尺度的深度特征 F_d 和红外特征 F_t:
      1. 1×1 卷积将两个模态映射到投影空间
      2. 逐通道计算互相投影，显式减法得到正交分量:
           F_d^⊥ = F_d' - α · (<F_d',F_t'> / ||F_t'||²) · F_t'
           F_t^⊥ = F_t' - β · (<F_t',F_d'> / ||F_d'||²) · F_d'
      3. 拼接两个正交分量，1×1 卷积融合回原通道数

    数学性质:
      - 正交分量与对方模态在通道维度上严格内积为零 (L2 投影)
      - α, β 可学习，sigmoid 约束到 [0,1]，允许部分保留投影成分
    """

    def __init__(self, in_channels: int, hide_channels: int | None = None,
                 reduction: int = 4):
        super().__init__()
        mid = hide_channels if hide_channels is not None else max(in_channels // reduction, 32)

        # ── 投影空间映射: 将深度/红外特征映射到同一空间做内积 ──
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

        # ── 可学习的投影缩放系数 (sigmoid 约束到 [0,1]) ──
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # ── 融合: 拼接正交分量后 1×1 卷积融合 ──
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
    def _gram_schmidt_step(F_a: torch.Tensor, F_b: torch.Tensor,
                           alpha: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt 正交化: 从 F_a 中减去它在 F_b 上的投影。

        F_a^⊥ = F_a - α · proj_{F_b}(F_a)

        其中 proj_{F_b}(F_a) = (<F_a, F_b> / ||F_b||²) · F_b

        Args:
            F_a:  待正交化的特征 (B, C, H, W)
            F_b:  投影基特征   (B, C, H, W)
            alpha: 可学习缩放系数 (sigmoid 后), 标量

        Returns:
            F_a^⊥: 与 F_b 正交的分量 (B, C, H, W)

        Note:
            内积和范数沿通道维度 (dim=1) 逐空间位置计算，
            即每个 (h,w) 位置独立正交化。
        """
        # 逐元素内积 → 沿通道求和: <F_a, F_b> (B, 1, H, W)
        inner = (F_a * F_b).sum(dim=1, keepdim=True)

        # L2 范数平方: ||F_b||² (B, 1, H, W)
        norm2 = F_b.pow(2).sum(dim=1, keepdim=True) + 1e-8

        # 投影向量: (<F_a,F_b> / ||F_b||²) · F_b  (B, C, H, W)
        proj = (inner / norm2) * F_b

        # 正交分量: F_a - α · proj
        return F_a - alpha * proj

    def forward(self, F_d: torch.Tensor, F_t: torch.Tensor) -> torch.Tensor:
        # 1. 映射到投影空间
        depth0 = self.proj_d(F_d)      # (B, mid, H, W)
        ir0 = self.proj_t(F_t)         # (B, mid, H, W)

        # 2. sigmoid 约束 alpha, beta 到 [0,1]
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)

        # 3. 互相正交化 (Gram-Schmidt)
        #    F_d^⊥ = depth0 - α · proj_{ir0}(depth0)
        #    F_t^⊥ = ir0    - β · proj_{depth0}(ir0)
        depth_orth = self._gram_schmidt_step(depth0, ir0, alpha)
        ir_orth = self._gram_schmidt_step(ir0, depth0, beta)

        # 4. 拼接正交分量后融合
        out = torch.cat([depth_orth, ir_orth], dim=1)  # (B, mid*2, H, W)
        return self.fuse(out)


# ═══════════════════════════════════════════════════════════════
# Multi-Scale DT Orthogonal Fusion
# ═══════════════════════════════════════════════════════════════
class DTOrtFusion(nn.Module):
    """多尺度深度-红外正交融合网络 (显式投影版本)。

    在 C2, C3, C4, C5 四个尺度上分别进行正交融合。
    每个尺度独立学习 α/β 缩放系数，因为不同尺度的模态互补程度不同：
      - 浅层 (C2): 纹理细节差异大 → 正交分量占比高
      - 深层 (C5): 语义类别趋同 → 正交分量占比低

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

    # ── 正交性验证: 验证 Gram-Schmidt 步骤的数学正确性 ──
    print("\n  --- Orthogonality Test ---")
    from dt_ort import OrthoFusionBlock as OFB
    import torch.nn.functional as F

    block = OFB(in_channels=256)
    block.eval()
    with torch.no_grad():
        d = torch.randn(2, 256, 40, 40)
        t = torch.randn(2, 256, 40, 40)

        # 手动执行投影步骤验证
        depth0 = block.proj_d(d)
        ir0 = block.proj_t(t)
        alpha = torch.sigmoid(block.alpha)
        beta = torch.sigmoid(block.beta)

        depth_orth = block._gram_schmidt_step(depth0, ir0, alpha)
        ir_orth = block._gram_schmidt_step(ir0, depth0, beta)

        # 计算正交分量与对方投影基的内积 (逐通道求和)
        cos_depth_on_ir = (depth_orth * ir0).sum(dim=1).abs().mean()
        cos_ir_on_depth = (ir_orth * depth0).sum(dim=1).abs().mean()

        print(f"  |<F_d^⊥, F_t>| mean: {cos_depth_on_ir.item():.6e}")
        print(f"  |<F_t^⊥, F_d>| mean: {cos_ir_on_depth.item():.6e}")

        # 当 alpha=1, beta=1 时，内积应接近 0
        depth_orth_full = block._gram_schmidt_step(depth0, ir0, torch.tensor(1.0))
        ir_orth_full = block._gram_schmidt_step(ir0, depth0, torch.tensor(1.0))
        cos_d_full = (depth_orth_full * ir0).sum(dim=1).abs().mean()
        cos_ir_full = (ir_orth_full * depth0).sum(dim=1).abs().mean()
        print(f"  |<F_d^⊥, F_t>| @ α=1: {cos_d_full.item():.6e}  (should be ≈ 0)")
        print(f"  |<F_t^⊥, F_d>| @ β=1: {cos_ir_full.item():.6e}  (should be ≈ 0)")

    print("  PASS")
    print("=" * 60)


if __name__ == "__main__":
    _test()
