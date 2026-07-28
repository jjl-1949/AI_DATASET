"""
UrbanDetector — 城市多目标识别主网络

三传感器 (RGB + Depth + Thermal) 端到端检测模型。

数据流:
    RGB (B,3,H,W)    → RGBBranch     → RGBFPN ─┐
                                                 ├→ DCR-CBAM → DMLab → DetHead
    Depth (B,1,H,W)  → DepthBranch   ─┐         │
                                       ├→ DTOrtFusion → DTFPN ─┘
    Thermal (B,1,H,W) → InfraredBranch ─┘

输出:
    {
        "cls_logits":  (B, num_classes, H/4, W/4),   # 类别 logits
        "bbox_preds":  (B, 4, H/4, W/4),             # 框距离 (l,t,r,b) > 0
        "centerness":  (B, 1, H/4, W/4),             # 中心度 ∈ [0,1]
        "attention":   {"P2": {"channel":..., "spatial":...}, ...}  # DCR-CBAM 注意力图
    }

使用示例:
    model = UrbanDetector(num_classes=12)
    rgb = torch.randn(2, 3, 360, 640)
    depth = torch.randn(2, 1, 360, 640)
    thermal = torch.randn(2, 1, 360, 640)
    output, attention = model(rgb, depth, thermal)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from rgb_deal import RGBBranch
from hw_deal import InfraredBranch
from dep_deal import DepthBranch
from dt_ort import DTOrtFusion
from rgb_fpn import RGBFPN
from dt_fpn import DTFPN
from fea_merge import MultiScaleDCRCBAM, channel_saliency_loss
from dmlab import DMLab
from det_head import DetHead


# ═══════════════════════════════════════════════════════════════
# 主网络
# ═══════════════════════════════════════════════════════════════
class UrbanDetector(nn.Module):
    """城市多目标识别三传感器端到端检测网络。

    模块组成:
        1. 三路 ResNet50 骨干 (RGB pretrained, Depth/IR from scratch)
        2. Depth-Thermal Gram-Schmidt 正交融合
        3. 双路 FPN (RGB + DT)
        4. DCR-CBAM 跨模态动态注意力融合
        5. DMLab 多级解码器 (ASPP + skip connections)
        6. FCOS Anchor-Free 检测头

    Args:
        num_classes:        类别数 (默认 12)
        rgb_frozen_stages:  RGB 骨干冻结阶段数 (默认 1: stem+layer1)
        fpn_channels:       FPN 输出通道数 (默认 256)
        reduction_mode:     DCR-CBAM 降维模式 ("low_rank" 或 "full")
        dynamic_rank:       low_rank 模式的秩 (默认 8)
        dmlab_skip_out:     DMLab skip 分支通道数 (默认 48)
        det_num_conv:       检测头每个子网的卷积层数 (默认 4)
    """

    def __init__(self,
                 num_classes: int = 12,
                 rgb_frozen_stages: int = 1,
                 fpn_channels: int = 256,
                 reduction_mode: str = "low_rank",
                 dynamic_rank: int = 8,
                 dmlab_skip_out: int = 48,
                 det_num_conv: int = 4):
        super().__init__()

        self.num_classes = num_classes

        # ══ 1. 三路骨干 ══
        self.rgb_branch = RGBBranch(frozen_stages=rgb_frozen_stages)
        self.depth_branch = DepthBranch()
        self.ir_branch = InfraredBranch()

        # ══ 2. Depth-Thermal 正交融合 ══
        self.dt_ort = DTOrtFusion()

        # ══ 3. 双路 FPN ══
        self.rgb_fpn = RGBFPN(out_channels=fpn_channels)
        self.dt_fpn = DTFPN(out_channels=fpn_channels)

        # ══ 4. DCR-CBAM 跨模态融合 ══
        self.dcr_cbam = MultiScaleDCRCBAM(
            channels=fpn_channels,
            reduction_mode=reduction_mode,
            dynamic_rank=dynamic_rank,
        )

        # ══ 5. DMLab 解码器 ══
        self.dmlab = DMLab(
            channels=fpn_channels,
            skip_out=dmlab_skip_out,
            out_channels=fpn_channels,
        )

        # ══ 6. 检测头 ══
        self.det_head = DetHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_conv=det_num_conv,
        )

    def forward(self,
                rgb: torch.Tensor,
                depth: torch.Tensor,
                thermal: torch.Tensor
                ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        """前向传播。

        Args:
            rgb:     RGB 图像    (B, 3, H, W)
            depth:   深度图      (B, 1, H, W)
            thermal: 红外热成像  (B, 1, H, W)

        Returns:
            detections: 检测输出字典
                - "cls_logits":  (B, num_classes, H/4, W/4)
                - "bbox_preds":  (B, 4, H/4, W/4)
                - "centerness":  (B, 1, H/4, W/4)
            attention: DCR-CBAM 注意力图 (用于 channel_saliency_loss)
                - {"P2": {"channel": (B,512,1,1), "spatial": (B,1,H/4,W/4)}, ...}
        """
        # ── RGB 通路 ──
        rgb_feats = self.rgb_branch(rgb)          # C2-C5: 256,512,1024,2048
        rgb_pyramid = self.rgb_fpn(rgb_feats)     # P2-P5: all 256

        # ── Depth-Thermal 通路 ──
        depth_feats = self.depth_branch(depth)
        ir_feats = self.ir_branch(thermal)
        dt_fused = self.dt_ort(depth_feats, ir_feats)  # C2-C5, channels preserved
        dt_pyramid = self.dt_fpn(dt_fused)             # P2-P5: all 256

        # ── 跨模态注意力融合 ──
        fused, attention = self.dcr_cbam(rgb_pyramid, dt_pyramid)

        # ── 解码 + 检测 ──
        features = self.dmlab(fused)              # (B, 256, H/4, W/4)
        detections = self.det_head(features)

        return detections, attention

    def compute_loss(self,
                     detections: Dict[str, torch.Tensor],
                     attention: Dict[str, Dict[str, torch.Tensor]],
                     targets: Dict[str, torch.Tensor],
                     saliency_weight: float = 1e-4
                     ) -> Dict[str, torch.Tensor]:
        """计算训练损失 (占位, 由用户根据实际 Loss 实现替换)。

        Args:
            detections:      forward() 返回的检测输出
            attention:       forward() 返回的注意力图
            targets:         训练标签 (格式由用户定义)
            saliency_weight: channel saliency 正则化权重

        Returns:
            losses: 各损失项字典 {"cls_loss":..., "reg_loss":..., "ctr_loss":..., "sal_loss":...}
        """
        # NOTE: 此方法为占位实现。实际训练时需根据标注格式实现:
        #   - Focal Loss (分类)
        #   - GIoU / L1 Loss (回归)
        #   - BCE Loss (中心度)
        #   - channel_saliency_loss (正则化)
        cls_loss = torch.tensor(0.0, device=detections["cls_logits"].device)
        reg_loss = torch.tensor(0.0, device=detections["cls_logits"].device)
        ctr_loss = torch.tensor(0.0, device=detections["cls_logits"].device)

        # 从 cls_logits 和 bbox_preds 构造 pseudo-loss (仅用于验证梯度流)
        cls_loss = cls_loss + detections["cls_logits"].abs().mean() * 0.0
        reg_loss = reg_loss + detections["bbox_preds"].abs().mean() * 0.0
        ctr_loss = ctr_loss + detections["centerness"].abs().mean() * 0.0

        sal_loss = channel_saliency_loss(attention) * saliency_weight

        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "ctr_loss": ctr_loss,
            "sal_loss": sal_loss,
            "total": cls_loss + reg_loss + ctr_loss + sal_loss,
        }


# ── 快速验证 ──────────────────────────────────────────────────
def _test():
    print("=" * 65)
    print("[UrbanDetector] Full Pipeline Test")
    B, H, W = 2, 640, 640
    num_classes = 12

    # ── 构建模型 ──
    print("\n[Building model...]")
    model = UrbanDetector(num_classes=num_classes)
    model.train()

    # ── 模拟输入 ──
    rgb = torch.randn(B, 3, H, W)
    depth = torch.randn(B, 1, H, W)
    thermal = torch.randn(B, 1, H, W)

    # ── 前向传播 ──
    print("\n[Forward pass]")
    detections, attention = model(rgb, depth, thermal)

    print(f"  RGB input:     {tuple(rgb.shape)}")
    print(f"  Depth input:   {tuple(depth.shape)}")
    print(f"  Thermal input: {tuple(thermal.shape)}")
    print(f"  Outputs:")
    for k, v in detections.items():
        print(f"    {k}: {tuple(v.shape)}")

    # ── 形状验证 ──
    assert detections["cls_logits"].shape == (B, num_classes, H // 4, W // 4)
    assert detections["bbox_preds"].shape == (B, 4, H // 4, W // 4)
    assert detections["centerness"].shape == (B, 1, H // 4, W // 4)
    assert detections["bbox_preds"].min() > 0, "bbox_preds must be > 0 (exp)"
    assert 0 <= detections["centerness"].min() <= detections["centerness"].max() <= 1

    for lv in ["P2", "P3", "P4", "P5"]:
        assert lv in attention, f"Missing attention level {lv}"

    print("\n  All shapes OK")

    # ── 梯度流验证 ──
    print("\n[Gradient flow]")
    losses = model.compute_loss(detections, attention, {})
    losses["total"].backward()

    total_params = 0
    all_ok = True
    for name in ["rgb_branch", "depth_branch", "ir_branch",
                 "dt_ort", "rgb_fpn", "dt_fpn", "dcr_cbam", "dmlab", "det_head"]:
        sub = getattr(model, name)
        grad_count = sum(1 for p in sub.parameters() if p.requires_grad and p.grad is not None)
        trainable = sum(1 for p in sub.parameters() if p.requires_grad)
        p = sum(p.numel() for p in sub.parameters())
        total_params += p
        ok = grad_count == trainable
        if not ok:
            all_ok = False
        print(f"  {name:20s}: {grad_count:>3}/{trainable:<3} gradients  "
              f"{p/1e6:6.1f}M  {'OK' if ok else 'MISSING!'}")

    print(f"\n  Gradient flow: {'ALL OK' if all_ok else 'ISSUES FOUND'}")
    print(f"  Total params:  {total_params/1e6:.1f}M")

    # ── 非正方形输入 ──
    print("\n[Non-square input 360x640]")
    rgb2 = torch.randn(B, 3, 360, 640)
    depth2 = torch.randn(B, 1, 360, 640)
    thermal2 = torch.randn(B, 1, 360, 640)
    with torch.no_grad():
        model.eval()
        det2, att2 = model(rgb2, depth2, thermal2)
    print(f"  cls_logits:  {tuple(det2['cls_logits'].shape)}")
    print(f"  bbox_preds:  {tuple(det2['bbox_preds'].shape)}")
    print(f"  centerness:  {tuple(det2['centerness'].shape)}")
    assert det2["cls_logits"].shape == (B, num_classes, 360 // 4, 640 // 4)
    print("  OK")

    # ── eval 模式一致性 ──
    print("\n[Train/Eval mode consistency]")
    model.train()
    det_train, _ = model(rgb, depth, thermal)
    model.eval()
    with torch.no_grad():
        det_eval, _ = model(rgb, depth, thermal)
    print(f"  train vs eval output difference: "
          f"cls={((det_train['cls_logits'] - det_eval['cls_logits']).abs().max()):.4f}, "
          f"box={((det_train['bbox_preds'] - det_eval['bbox_preds']).abs().max()):.4f}")
    print(f"  (BN causes small differences — expected)")

    print("\n  PASS")
    print("=" * 65)


if __name__ == "__main__":
    _test()
