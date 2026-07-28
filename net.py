"""
UrbanDetector — 城市多目标识别主网络

三传感器 (RGB + Depth + Thermal) 端到端检测模型。

数据流:
    RGB (B,3,H,W)    → RGBBranch     → RGBFPN ─┐
                                                 ├→ DCR-CBAM → DMLab → DetHead
    Depth (B,1,H,W)  → DepthBranch   ─┐         │
                                       ├→ DTOrtFusion → DTFPN ─┘
    Thermal (B,1,H,W) → InfraredBranch ─┘

损失函数 (论文公式 12):
    L_total = lambda1*L_label + lambda2*L_bbox + lambda3*L_giou + heta_Mc + heta_Ms

    L_label:  Focal Loss (alpha=0.25, gamma=2.0) — 解决类别不平衡
    L_bbox:   L1 Loss — 边框回归精度
    L_giou:   GIoU Loss — 边框空间对齐
    heta_Mc:  Channel Saliency (L1) — 通道注意力稀疏正则化
    heta_Ms:  Spatial Focus (L2) — 空间注意力目标聚焦正则化

使用示例:
    model = UrbanDetector(num_classes=12)
    rgb = torch.randn(2, 3, 360, 640)
    depth = torch.randn(2, 1, 360, 640)
    thermal = torch.randn(2, 1, 360, 640)
    detections, attention = model(rgb, depth, thermal)
    losses = model.compute_loss(detections, attention, gt_boxes, gt_labels, (360,640))
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
from dmlab import DMLab
from det_head import DetHead


# ═══════════════════════════════════════════════════════════════
# Spatial Saliency Loss (to be moved to fea_merge.py)
# ═══════════════════════════════════════════════════════════════

def _build_gt_mask(gt_boxes: torch.Tensor, img_h: int, img_w: int,
                   h_k: int, w_k: int, device: torch.device) -> torch.Tensor:
    """Create binary mask from GT boxes, interpolated to feature level size."""
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
    """heta_Ms: target-focusing spatial regularizer (paper eq.).

    For each level k:
        B_mask = binary GT mask interpolated to Ms_k spatial size
        residual = (Ms_k * B_mask) - B_mask
        loss_k = ||residual||_2^2

    Inside boxes: pushes Ms -> 1  (attend to objects)
    Outside boxes: 0 penalty (mask = 0)
    """
    img_h, img_w = image_size
    batch_size = len(gt_boxes)
    device = next(iter(attention.values()))["spatial"].device
    total = torch.tensor(0.0, device=device)
    for maps in attention.values():
        ms = maps["spatial"]                      # (B, 1, H_k, W_k)
        _, _, h_k, w_k = ms.shape
        level_loss = torch.tensor(0.0, device=device)
        for b in range(batch_size):
            if gt_boxes[b].numel() == 0:
                continue
            mask_interp = _build_gt_mask(gt_boxes[b], img_h, img_w,
                                         h_k, w_k, device)
            residual = ms[b, 0] * mask_interp - mask_interp
            level_loss = level_loss + (residual ** 2).mean()
        total = total + level_loss / max(batch_size, 1)
    return total


# ═══════════════════════════════════════════════════════════════
# Loss Utilities
# ═══════════════════════════════════════════════════════════════

def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal Loss for binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: (N, C) predicted logits
        targets: (N, C) one-hot targets (0 or 1)
        alpha:   weighting factor (default 0.25)
        gamma:   focusing parameter (default 2.0)
    """
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = targets * prob + (1 - targets) * (1 - prob)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * ce).mean()


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 loss for bounding box regression."""
    return F.smooth_l1_loss(pred, target, reduction="mean")


def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Generalized IoU loss.

    GIoU = IoU - |C minus (A union B)| / |C|,  Loss = 1 - GIoU

    Args:
        pred_boxes:   (N, 4) in (x1, y1, x2, y2) format
        target_boxes: (N, 4) in (x1, y1, x2, y2) format
    """
    # Intersection
    ix1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    iy1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    ix2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    iy2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    # Union
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_target = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union = area_pred + area_target - inter + 1e-7
    iou = inter / union

    # Smallest enclosing box
    cx1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    cy1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    cx2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    cy2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    c_area = (cx2 - cx1) * (cy2 - cy1) + 1e-7

    giou = iou - (c_area - union) / c_area
    return (1 - giou).mean()


def fcos_targets(gt_boxes: List[torch.Tensor], gt_labels: List[torch.Tensor],
                 feat_h: int, feat_w: int, stride: int,
                 num_classes: int, device: torch.device):
    """Assign FCOS targets to each feature map location.

    Each location (i,j) maps to image point (s*i+s/2, s*j+s/2).
    If inside a GT box, assigned to it.

    Args:
        gt_boxes:  list of (N_b, 4) in (x1,y1,x2,y2) pixel coords, per batch
        gt_labels: list of (N_b,) class ids, per batch
        feat_h:    feature map height
        feat_w:    feature map width
        stride:    feature stride (4)
        num_classes: number of classes
        device:    torch device

    Returns:
        cls_target:  (B, num_classes, feat_h, feat_w) one-hot
        reg_target:  (B, 4, feat_h, feat_w) (l,t,r,b) distances
        ctr_target:  (B, 1, feat_h, feat_w) centerness
        mask:        (B, 1, feat_h, feat_w) 1=positive, 0=ignore
    """
    B = len(gt_boxes)
    cls_target = torch.zeros(B, num_classes, feat_h, feat_w, device=device)
    reg_target = torch.zeros(B, 4, feat_h, feat_w, device=device)
    ctr_target = torch.zeros(B, 1, feat_h, feat_w, device=device)
    mask = torch.zeros(B, 1, feat_h, feat_w, device=device)

    for b in range(B):
        boxes = gt_boxes[b]    # (N, 4) in (x1,y1,x2,y2)
        labels = gt_labels[b]  # (N,)
        if boxes.numel() == 0:
            continue

        for n in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[n]
            cls_id = int(labels[n].item())

            # Map box to feature grid indices (float)
            gx1 = x1 / stride; gy1 = y1 / stride
            gx2 = x2 / stride; gy2 = y2 / stride

            # Integer grid cells inside this box
            ix1 = max(0, int(gx1)); iy1 = max(0, int(gy1))
            ix2 = min(feat_w, int(gx2) + 1); iy2 = min(feat_h, int(gy2) + 1)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            # For each location in the box
            for iy in range(iy1, iy2):
                for ix in range(ix1, ix2):
                    # Skip if already assigned (first box wins)
                    if mask[b, 0, iy, ix] > 0:
                        continue

                    # Image coordinate of this location
                    loc_x = stride * ix + stride / 2.0
                    loc_y = stride * iy + stride / 2.0

                    # Distances to box edges
                    l = loc_x - x1
                    t = loc_y - y1
                    r = x2 - loc_x
                    b_bot = y2 - loc_y

                    # Only assign if inside box (with tolerance)
                    if l <= 0 or t <= 0 or r <= 0 or b_bot <= 0:
                        continue

                    cls_target[b, cls_id, iy, ix] = 1.0
                    reg_target[b, 0, iy, ix] = l
                    reg_target[b, 1, iy, ix] = t
                    reg_target[b, 2, iy, ix] = r
                    reg_target[b, 3, iy, ix] = b_bot

                    # Centerness = sqrt(min(l,r)*min(t,b) / max(l,r)*max(t,b))
                    ctr = ((min(l, r) * min(t, b_bot)) /
                           (max(l, r) * max(t, b_bot) + 1e-7)) ** 0.5
                    ctr_target[b, 0, iy, ix] = ctr
                    mask[b, 0, iy, ix] = 1.0

    return cls_target, reg_target, ctr_target, mask


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

        # ── 损失权重 (论文公式 12) ──
        self.lambda_cls = 1.0    # lambda1: Focal Loss
        self.lambda_l1 = 1.0     # lambda2: L1 Loss
        self.lambda_giou = 2.0   # lambda3: GIoU Loss
        self.lambda_mc = 1e-4    # Channel saliency
        self.lambda_ms = 1e-4    # Spatial saliency

    def forward(self,
                rgb: torch.Tensor,
                depth: torch.Tensor,
                thermal: torch.Tensor
                ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        """前向传播。"""
        # ── RGB 通路 ──
        rgb_feats = self.rgb_branch(rgb)
        rgb_pyramid = self.rgb_fpn(rgb_feats)

        # ── Depth-Thermal 通路 ──
        depth_feats = self.depth_branch(depth)
        ir_feats = self.ir_branch(thermal)
        dt_fused = self.dt_ort(depth_feats, ir_feats)
        dt_pyramid = self.dt_fpn(dt_fused)

        # ── 跨模态注意力融合 ──
        fused, attention = self.dcr_cbam(rgb_pyramid, dt_pyramid)

        # ── 解码 + 检测 ──
        features = self.dmlab(fused)
        detections = self.det_head(features)

        return detections, attention

    def compute_loss(self,
                     detections: Dict[str, torch.Tensor],
                     attention: Dict[str, Dict[str, torch.Tensor]],
                     gt_boxes: List[torch.Tensor],
                     gt_labels: List[torch.Tensor],
                     image_size: Tuple[int, int],
                     ) -> Dict[str, torch.Tensor]:
        """计算训练损失 (论文公式 12)。

        L_total = lambda1*L_label + lambda2*L_bbox + lambda3*L_giou
                  + lambda_mc*heta_Mc + lambda_ms*heta_Ms

        Args:
            detections:  forward() 返回的检测输出
            attention:   DCR-CBAM 注意力图
            gt_boxes:    list of (N_b, 4) GT boxes in (x1,y1,x2,y2) pixel coords
            gt_labels:   list of (N_b,)   GT class ids
            image_size:  (H, W) of input image

        Returns:
            losses dict with "cls_loss", "l1_loss", "giou_loss",
            "mc_loss", "ms_loss", "total"
        """
        device = detections["cls_logits"].device
        B, C, H_f, W_f = detections["cls_logits"].shape

        # ── 1. FCOS target assignment ──
        cls_target, reg_target, ctr_target, pos_mask = fcos_targets(
            gt_boxes, gt_labels, H_f, W_f, stride=4,
            num_classes=self.num_classes, device=device)

        pos = pos_mask.bool()                     # (B, 1, H_f, W_f)
        num_pos = max(pos.sum(), 1)

        # ── 2. Classification Loss (Focal Loss) ──
        cls_logits = detections["cls_logits"]     # (B, C, H_f, W_f)
        cls_logits_flat = cls_logits.permute(0, 2, 3, 1).reshape(-1, C)
        cls_target_flat = cls_target.permute(0, 2, 3, 1).reshape(-1, C)
        L_label = focal_loss(cls_logits_flat, cls_target_flat)

        # ── 3. Regression Loss (L1 + GIoU) ──
        bbox_preds = detections["bbox_preds"]     # (B, 4, H_f, W_f)  (l,t,r,b)

        # Select positive locations
        pos_indices = pos.squeeze(1).nonzero(as_tuple=False)  # (K, 3): (b, y, x)
        if pos_indices.numel() > 0:
            pred_ltrb = bbox_preds[pos_indices[:, 0], :, pos_indices[:, 1], pos_indices[:, 2]]
            target_ltrb = reg_target[pos_indices[:, 0], :, pos_indices[:, 1], pos_indices[:, 2]]

            # Convert FCOS (l,t,r,b) to (x1,y1,x2,y2) for GIoU
            img_h, img_w = image_size
            loc_x = 4.0 * pos_indices[:, 2].float() + 2.0   # stride * x + stride/2
            loc_y = 4.0 * pos_indices[:, 1].float() + 2.0

            pred_x1 = loc_x - pred_ltrb[:, 0]; pred_y1 = loc_y - pred_ltrb[:, 1]
            pred_x2 = loc_x + pred_ltrb[:, 2]; pred_y2 = loc_y + pred_ltrb[:, 3]
            pred_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

            target_x1 = loc_x - target_ltrb[:, 0]; target_y1 = loc_y - target_ltrb[:, 1]
            target_x2 = loc_x + target_ltrb[:, 2]; target_y2 = loc_y + target_ltrb[:, 3]
            target_xyxy = torch.stack([target_x1, target_y1, target_x2, target_y2], dim=1)

            L_bbox = l1_loss(pred_ltrb, target_ltrb)
            L_giou = giou_loss(pred_xyxy, target_xyxy)
        else:
            L_bbox = torch.tensor(0.0, device=device)
            L_giou = torch.tensor(0.0, device=device)

        # ── 4. Centerness Loss (BCE) ──
        ctr_preds = detections["centerness"].squeeze(1)         # (B, H_f, W_f)
        ctr_target_sq = ctr_target.squeeze(1)                   # (B, H_f, W_f)
        if pos.sum() > 0:
            L_ctr = F.binary_cross_entropy(ctr_preds[pos.squeeze(1)],
                                            ctr_target_sq[pos.squeeze(1)])
        else:
            L_ctr = torch.tensor(0.0, device=device)

        # ── 5. DCR-CBAM Regularization ──
        L_mc = channel_saliency_loss(attention)                 # heta_Mc

        L_ms = spatial_saliency_loss(attention, gt_boxes, image_size)  # heta_Ms

        # ── Total ──
        total = (self.lambda_cls * L_label +
                 self.lambda_l1 * L_bbox +
                 self.lambda_giou * L_giou +
                 L_ctr +
                 self.lambda_mc * L_mc +
                 self.lambda_ms * L_ms)

        return {
            "cls_loss": L_label,
            "l1_loss": L_bbox,
            "giou_loss": L_giou,
            "ctr_loss": L_ctr,
            "mc_loss": L_mc,
            "ms_loss": L_ms,
            "total": total,
        }

    def set_loss_weights(self, lambda_cls=1.0, lambda_l1=1.0, lambda_giou=2.0,
                         lambda_mc=1e-4, lambda_ms=1e-4):
        """动态调整损失权重。"""
        self.lambda_cls = lambda_cls
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_mc = lambda_mc
        self.lambda_ms = lambda_ms


# ── 快速验证 ──────────────────────────────────────────────────
def _test():
    print("=" * 65)
    print("[UrbanDetector] Full Pipeline + Loss Test")
    B, H, W = 2, 640, 640
    num_classes = 12

    model = UrbanDetector(num_classes=num_classes)
    model.train()

    rgb = torch.randn(B, 3, H, W)
    depth = torch.randn(B, 1, H, W)
    thermal = torch.randn(B, 1, H, W)

    # ── 模拟 GT 标注 ──
    # 每张图 3 个物体
    gt_boxes = [
        torch.tensor([[100., 80.,  200., 300.], [300., 150., 500., 400.],
                      [50.,  200., 180., 500.]]),  # batch 0
        torch.tensor([[120., 60.,  250., 280.], [400., 100., 550., 350.],
                      [30.,  150., 160., 450.]]),  # batch 1
    ]
    gt_labels = [
        torch.tensor([0, 3, 6]),   # batch 0
        torch.tensor([2, 5, 8]),   # batch 1
    ]

    # ── 前向 ──
    print("\n[Forward pass]")
    detections, attention = model(rgb, depth, thermal)
    for k, v in detections.items():
        print(f"  {k}: {tuple(v.shape)}")

    # ── 计算损失 ──
    print("\n[Loss computation]")
    losses = model.compute_loss(detections, attention, gt_boxes, gt_labels, (H, W))
    for k, v in losses.items():
        print(f"  {k:12s}: {v.item():.6f}")

    assert losses["total"].item() > 0, "Total loss should be > 0"
    assert losses["cls_loss"].item() > 0, "Focal loss should be > 0"
    assert losses["l1_loss"].item() >= 0, "L1 loss should be >= 0"
    assert losses["giou_loss"].item() >= 0, "GIoU loss should be >= 0"

    # ── 梯度流 ──
    print("\n[Gradient flow through full loss]")
    model.zero_grad()
    losses["total"].backward()

    all_ok = True
    total_params = 0
    for name in ["rgb_branch", "depth_branch", "ir_branch",
                 "dt_ort", "rgb_fpn", "dt_fpn", "dcr_cbam", "dmlab", "det_head"]:
        sub = getattr(model, name)
        g = sum(1 for p in sub.parameters() if p.requires_grad and p.grad is not None)
        t = sum(1 for p in sub.parameters() if p.requires_grad)
        p = sum(p.numel() for p in sub.parameters())
        total_params += p
        ok = g == t
        if not ok: all_ok = False
        print(f"  {name:20s}: {g:>3}/{t:<3}  {p/1e6:6.1f}M  {'OK' if ok else 'MISSING!'}")

    print(f"\n  Gradient flow: {'ALL OK' if all_ok else 'ISSUES FOUND'}")
    print(f"  Total params:  {total_params/1e6:.1f}M")

    # ── 非正方形 + 空标注 ──
    print("\n[Edge cases]")
    model.eval()
    with torch.no_grad():
        d2, a2 = model(torch.randn(1, 3, 360, 640),
                       torch.randn(1, 1, 360, 640),
                       torch.randn(1, 1, 360, 640))
    print(f"  360x640: cls={tuple(d2['cls_logits'].shape)} OK")

    # 空标注 (no objects)
    losses_empty = model.compute_loss(d2, a2,
                                       [torch.zeros(0, 4)], [torch.zeros(0, dtype=torch.long)],
                                       (360, 640))
    print(f"  Empty GT: total={losses_empty['total'].item():.6f} (should be near 0 or saliency only)")

    print("\n  PASS")
    print("=" * 65)


if __name__ == "__main__":
    _test()
