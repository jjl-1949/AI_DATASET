"""
UrbanDetector — 城市多目标识别主网络

三传感器 (RGB + Depth + Thermal) 端到端检测模型。

数据流:
    RGB (B,3,H,W)    → RGBBranch     → RGBFPN ─┐
                                                 ├→ DCR-CBAM (P3-P7) → MultiScaleDetHead
    Depth (B,1,H,W)  → DepthBranch   ─┐         │
                                       ├→ DTOrtFusion → DTFPN ─┘
    Thermal (B,1,H,W) → InfraredBranch ─┘

FPN 输出 P2-P5，然后生成 P6, P7 (stride=64, 128) 用于大目标检测。

损失函数:
    L_total = lambda_cls*L_focal + lambda_l1*L_l1 + lambda_giou*L_giou
              + L_ctr + lambda_mc*heta_Mc + lambda_ms*heta_Ms

    L_focal:  Focal Loss (sum/max(num_pos,1)), center sampling
    L_l1:     Smooth L1 (sum/max(num_pos,1))
    L_giou:   GIoU Loss
    L_ctr:    BCE centerness
    heta_Mc:  Channel saliency L1
    heta_Ms:  Spatial focus L2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

from rgb_deal import RGBBranch
from hw_deal import InfraredBranch
from dep_deal import DepthBranch
from dt_ort import DTOrtFusion
from rgb_fpn import RGBFPN
from dt_fpn import DTFPN
from fea_merge import MultiScaleDCRCBAM, channel_saliency_loss
from det_head import MultiScaleDetHead, compute_det_loss


# ═══════════════════════════════════════════════════════════════
# Spatial Saliency Loss (to be moved to fea_merge.py when unlocked)
# ═══════════════════════════════════════════════════════════════

def _build_gt_mask(gt_boxes, img_h, img_w, h_k, w_k, device):
    mask = torch.zeros(img_h, img_w, device=device)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        x1c, y1c = max(0, int(x1)), max(0, int(y1))
        x2c, y2c = min(img_w, int(x2) + 1), min(img_h, int(y2) + 1)
        if x2c > x1c and y2c > y1c:
            mask[y1c:y2c, x1c:x2c] = 1.0
    return F.interpolate(mask[None, None, :, :], size=(h_k, w_k),
                         mode="bilinear", align_corners=False).squeeze(0).squeeze(0)


def spatial_saliency_loss(attention, gt_boxes, image_size):
    """heta_Ms: target-focusing spatial regularizer.
    For each level: residual = (Ms * B_mask) - B_mask, loss = ||residual||_2^2."""
    img_h, img_w = image_size
    batch_size = len(gt_boxes)
    device = next(iter(attention.values()))["spatial"].device
    total = torch.tensor(0.0, device=device)
    for maps in attention.values():
        ms = maps["spatial"]
        _, _, h_k, w_k = ms.shape
        level_loss = torch.tensor(0.0, device=device)
        for b in range(batch_size):
            if gt_boxes[b].numel() == 0:
                continue
            mask_interp = _build_gt_mask(gt_boxes[b], img_h, img_w, h_k, w_k, device)
            residual = ms[b, 0] * mask_interp - mask_interp
            level_loss = level_loss + (residual ** 2).mean()
        total = total + level_loss / max(batch_size, 1)
    return total


# ═══════════════════════════════════════════════════════════════
# 主网络
# ═══════════════════════════════════════════════════════════════
class UrbanDetector(nn.Module):
    """城市多目标识别三传感器端到端检测网络。

    模块组成:
        1. 三路 ResNet50 骨干 (RGB pretrained, Depth/IR from scratch)
        2. Depth-Thermal 严格正交融合 + 后门控
        3. 双路 FPN (RGB + DT) → P2-P5
        4. P6/P7 扩展 (stride 64, 128) — 注册的 conv 层
        5. DCR-CBAM 逐尺度跨模态注意力融合 (P3-P7)
        6. 多尺度 FCOS 检测头 (P3-P7, center sampling)
    """

    def __init__(self,
                 num_classes: int = 12,
                 rgb_frozen_stages: int = 1,
                 fpn_channels: int = 256,
                 reduction_mode: str = "low_rank",
                 dynamic_rank: int = 8,
                 det_num_conv: int = 4):
        super().__init__()

        self.num_classes = num_classes
        self.fpn_channels = fpn_channels

        # ══ 1. 三路骨干 ══
        self.rgb_branch = RGBBranch(frozen_stages=rgb_frozen_stages)
        self.depth_branch = DepthBranch()
        self.ir_branch = InfraredBranch()

        # ══ 2. Depth-Thermal 正交融合 ══
        self.dt_ort = DTOrtFusion()

        # ══ 3. 双路 FPN ══
        self.rgb_fpn = RGBFPN(out_channels=fpn_channels)
        self.dt_fpn = DTFPN(out_channels=fpn_channels)

        # ══ 4. P6/P7 生成 (从 P5 下采样) ══
        # 注册为模块参数以正确接收梯度
        self.rgb_p6_conv = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3,
                                      stride=2, padding=1, bias=False)
        self.rgb_p7_conv = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3,
                                      stride=2, padding=1, bias=False)
        self.dt_p6_conv = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3,
                                     stride=2, padding=1, bias=False)
        self.dt_p7_conv = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3,
                                     stride=2, padding=1, bias=False)
        for conv in [self.rgb_p6_conv, self.rgb_p7_conv,
                     self.dt_p6_conv, self.dt_p7_conv]:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")

        # ══ 5. DCR-CBAM 跨模态融合 (P2-P7, 所有 FPN 层级参与) ══
        dcr_levels = ("P2", "P3", "P4", "P5", "P6", "P7")
        self.dcr_levels = dcr_levels
        self.dcr_cbam = MultiScaleDCRCBAM(
            channels=fpn_channels,
            levels=dcr_levels,
            reduction_mode=reduction_mode,
            dynamic_rank=dynamic_rank,
        )

        # ══ 6. 多尺度检测头 ══
        self.det_head = MultiScaleDetHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_conv=det_num_conv,
            levels=dcr_levels,
        )

        # ── 损失权重 ──
        self.lambda_cls = 1.0
        self.lambda_l1 = 1.0
        self.lambda_giou = 2.0
        self.lambda_mc = 1e-4
        self.lambda_ms = 1e-4

    def _make_extra_levels(self,
                           fpn_feats: Dict[str, torch.Tensor],
                           p6_conv: nn.Conv2d,
                           p7_conv: nn.Conv2d) -> Dict[str, torch.Tensor]:
        """从 P5 生成 P6 (stride 64) 和 P7 (stride 128)。"""
        p5 = fpn_feats["P5"]
        p6 = p6_conv(p5)
        p7 = p7_conv(F.relu(p6))
        result = dict(fpn_feats)
        result["P6"] = p6
        result["P7"] = p7
        return result

    def forward(self,
                rgb: torch.Tensor,
                depth: torch.Tensor,
                thermal: torch.Tensor
                ) -> Tuple[Dict, Dict]:
        """前向传播。

        Args:
            rgb:     (B, 3, H, W)
            depth:   (B, 1, H, W)
            thermal: (B, 1, H, W)

        Returns:
            detections: {level: {"cls_logits":..., "bbox_preds":..., "centerness":...}}
            attention:  DCR-CBAM 注意力图
        """
        # ── RGB 通路 ──
        rgb_feats = self.rgb_branch(rgb)
        rgb_pyramid = self.rgb_fpn(rgb_feats)
        rgb_pyramid = self._make_extra_levels(rgb_pyramid, self.rgb_p6_conv, self.rgb_p7_conv)

        # ── Depth-Thermal 通路 ──
        depth_feats = self.depth_branch(depth)
        ir_feats = self.ir_branch(thermal)
        dt_fused = self.dt_ort(depth_feats, ir_feats)
        dt_pyramid = self.dt_fpn(dt_fused)
        dt_pyramid = self._make_extra_levels(dt_pyramid, self.dt_p6_conv, self.dt_p7_conv)

        # ── 跨模态注意力融合 (P3-P7) ──
        rgb_for_dcr = {k: v for k, v in rgb_pyramid.items() if k in self.dcr_levels}
        dt_for_dcr = {k: v for k, v in dt_pyramid.items() if k in self.dcr_levels}
        fused, attention = self.dcr_cbam(rgb_for_dcr, dt_for_dcr)

        # ── 多尺度检测 ──
        detections = self.det_head(fused)

        return detections, attention

    def compute_loss(self,
                     detections: Dict,
                     attention: Dict,
                     gt_boxes: List[torch.Tensor],
                     gt_labels: List[torch.Tensor],
                     image_size: Tuple[int, int],
                     ) -> Dict[str, torch.Tensor]:
        """计算训练损失。

        Args:
            detections:  forward() 检测输出
            attention:   DCR-CBAM 注意力图
            gt_boxes:    list of (N_b, 4) in (x1,y1,x2,y2) pixel coords
            gt_labels:   list of (N_b,) class ids
            image_size:  (H, W)

        Returns:
            {"cls_loss", "l1_loss", "giou_loss", "ctr_loss",
             "mc_loss", "ms_loss", "total", "num_pos"}
        """
        # ── 检测损失 (多尺度 FCOS) ──
        losses = compute_det_loss(
            detections, gt_boxes, gt_labels,
            self.det_head.strides, self.det_head.scale_ranges,
            self.num_classes,
            self.lambda_cls, self.lambda_l1, self.lambda_giou)

        # ── DCR-CBAM 正则化 ──
        L_mc = channel_saliency_loss(attention)
        L_ms = spatial_saliency_loss(attention, gt_boxes, image_size)

        losses["mc_loss"] = L_mc
        losses["ms_loss"] = L_ms
        losses["total"] = (losses["total"] +
                           self.lambda_mc * L_mc +
                           self.lambda_ms * L_ms)

        return losses

    def set_loss_weights(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ── 快速验证 ──────────────────────────────────────────────────
def _test():
    print("=" * 65)
    print("[UrbanDetector] Multi-Scale Pipeline + Loss Test")
    B, H, W = 2, 640, 640

    model = UrbanDetector(num_classes=12)
    model.train()

    rgb = torch.randn(B, 3, H, W)
    depth = torch.randn(B, 1, H, W)
    thermal = torch.randn(B, 1, H, W)

    # ── GT ──
    gt_boxes = [
        torch.tensor([[50., 40., 150., 200.], [100., 50., 500., 500.]]),
        torch.tensor([[80., 60., 200., 180.], [200., 100., 400., 500.]]),
    ]
    gt_labels = [
        torch.tensor([0, 3]),
        torch.tensor([2, 5]),
    ]

    # ── 前向 ──
    print("\n[Forward pass — per-level outputs]")
    detections, attention = model(rgb, depth, thermal)
    for lv, d in detections.items():
        print(f"  {lv}: cls={tuple(d['cls_logits'].shape)}, "
              f"reg={tuple(d['bbox_preds'].shape)}, "
              f"ctr={tuple(d['centerness'].shape)}")

    # ── 损失 ──
    print("\n[Loss computation]")
    losses = model.compute_loss(detections, attention, gt_boxes, gt_labels, (H, W))
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:12s}: {v.item():.6f}")
        else:
            print(f"  {k:12s}: {v}")

    assert losses["total"].item() > 0
    assert losses["num_pos"] > 0
    assert losses["ms_loss"].item() > 0

    # ── 梯度流 ──
    print("\n[Gradient flow]")
    model.zero_grad()
    losses["total"].backward()

    all_ok = True
    total_p = 0
    for name in ["rgb_branch", "depth_branch", "ir_branch",
                 "dt_ort", "rgb_fpn", "dt_fpn", "dcr_cbam", "det_head"]:
        sub = getattr(model, name)
        g = sum(1 for p in sub.parameters() if p.requires_grad and p.grad is not None)
        t = sum(1 for p in sub.parameters() if p.requires_grad)
        p = sum(p.numel() for p in sub.parameters())
        total_p += p
        ok = g == t
        if not ok: all_ok = False
        print(f"  {name:20s}: {g:>3}/{t:<3}  {p/1e6:6.1f}M  {'OK' if ok else 'MISSING!'}")

    print(f"\n  Gradient flow: {'ALL OK' if all_ok else 'ISSUES FOUND'}")
    print(f"  Total params:  {total_p/1e6:.1f}M")

    # ── 非正方形 ──
    print("\n[360x640 input]")
    model.eval()
    with torch.no_grad():
        d2, _ = model(torch.randn(1, 3, 360, 640),
                      torch.randn(1, 1, 360, 640),
                      torch.randn(1, 1, 360, 640))
    for lv in ["P3", "P5", "P7"]:
        print(f"  {lv}: cls={tuple(d2[lv]['cls_logits'].shape)}")
    print("  OK")

    print("\n  PASS")
    print("=" * 65)


if __name__ == "__main__":
    _test()
