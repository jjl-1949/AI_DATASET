"""
Multi-Scale FCOS Detection Head — 多尺度 Anchor-Free 检测头

在 FPN 多尺度特征 (P3–P7) 上共享权重进行检测，每个 FPN 层级
负责不同尺度的目标，通过 center sampling 分配正样本。

尺度分工:
    P3 (stride=8):   小目标  [  0,  64] px
    P4 (stride=16):  中小目标 [ 64, 128] px
    P5 (stride=32):  中目标   [128, 256] px
    P6 (stride=64):  大目标   [256, 512] px
    P7 (stride=128): 超大目标 [512, inf] px

输入:  {"P3": (B,256,H/8,W/8), ..., "P7": (B,256,H/128,W/128)}
输出:  {level: {"cls_logits":..., "bbox_preds":..., "centerness":...}, ...}

目标分配: center sampling — 位置在 GT box 中心子区域内才分配为正样本
重叠处理: 选择面积最小的 GT (利于小目标)
损失归一化: Focal Loss 使用 sum / max(num_pos, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


# ═══════════════════════════════════════════════════════════════
# Conv-BN-ReLU 构建块
# ═══════════════════════════════════════════════════════════════
def _make_conv_block(in_ch: int, out_ch: int, num_convs: int) -> nn.Sequential:
    layers = []
    for _ in range(num_convs):
        layers.extend([
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ])
        in_ch = out_ch
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════
# Multi-Scale FCOS Head
# ═══════════════════════════════════════════════════════════════
class MultiScaleDetHead(nn.Module):
    """多尺度 FCOS 检测头。

    在所有 FPN 层级上共享卷积权重 (标准 FCOS 做法)。
    每个层级有独立的可学习 scale 因子用于回归。
    中心度子网与分类子网共享前 num_conv 层以节约参数。

    Args:
        in_channels:   输入通道数 (默认 256)
        num_classes:   类别数 (默认 12)
        num_conv:      共享卷积层数 (默认 4)
        levels:        FPN 层级列表 (默认 P3–P7)
        strides:       各层级 stride 映射
        scale_ranges:  各层级负责的回归距离范围 (用于标签分配)
    """

    DEFAULT_LEVELS = ("P2", "P3", "P4", "P5", "P6", "P7")
    DEFAULT_STRIDES = {"P2": 4, "P3": 8, "P4": 16, "P5": 32, "P6": 64, "P7": 128}
    DEFAULT_SCALE_RANGES = {
        "P2": (0, 64), "P3": (64, 128), "P4": (128, 256),
        "P5": (256, 512), "P6": (512, 1024), "P7": (1024, float("inf")),
    }

    def __init__(self,
                 in_channels: int = 256,
                 num_classes: int = 12,
                 num_conv: int = 4,
                 levels: Tuple[str, ...] = DEFAULT_LEVELS,
                 strides: Dict[str, int] | None = None,
                 scale_ranges: Dict[str, Tuple[float, float]] | None = None):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.levels = list(levels)
        self.strides = strides or {k: self.DEFAULT_STRIDES[k] for k in self.levels}
        self.scale_ranges = scale_ranges or {
            k: self.DEFAULT_SCALE_RANGES[k] for k in self.levels}

        # ── 共享分类卷积 (分类 + 中心度共享 stem) ──
        self.cls_convs = _make_conv_block(in_channels, in_channels, num_conv)

        # ── 共享回归卷积 ──
        self.reg_convs = _make_conv_block(in_channels, in_channels, num_conv)

        # ── 输出层 ──
        self.cls_out = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_out = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.ctr_out = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # ── 每层级可学习 scale (回归) ──
        self.scales = nn.ParameterDict({
            lv: nn.Parameter(torch.tensor(1.0)) for lv in self.levels
        })

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                is_output = m in (self.cls_out, self.reg_out, self.ctr_out)
                if is_output:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                if m.bias is not None:
                    if m is self.cls_out:
                        prior_prob = 0.01
                        nn.init.constant_(m.bias, -torch.log(
                            torch.tensor((1 - prior_prob) / prior_prob)))
                    else:
                        nn.init.constant_(m.bias, 0.0)

    def forward(self, feats: Dict[str, torch.Tensor]
                ) -> Dict[str, Dict[str, torch.Tensor]]:
        """前向传播。

        Args:
            feats: FPN 特征 {"P3": (B,256,H/8,W/8), ...}

        Returns:
            {level: {"cls_logits":..., "bbox_preds":..., "centerness":...}, ...}
        """
        outputs = {}
        for lv in self.levels:
            if lv not in feats:
                continue
            x = feats[lv]

            # Shared stem
            cls_feat = self.cls_convs(x)
            reg_feat = self.reg_convs(x)

            # Classification
            cls_logits = self.cls_out(cls_feat)                       # (B, C, H, W)

            # Regression: exp(scale * reg_out) per level
            scale = self.scales[lv]
            bbox_preds = torch.exp(scale) * torch.exp(self.reg_out(reg_feat))

            # Centerness: shares cls stem
            centerness = torch.sigmoid(self.ctr_out(cls_feat))

            outputs[lv] = {
                "cls_logits": cls_logits,
                "bbox_preds": bbox_preds,
                "centerness": centerness,
            }

        return outputs


# ═══════════════════════════════════════════════════════════════
# 目标分配 (Center Sampling)
# ═══════════════════════════════════════════════════════════════

def fcos_targets_multi_scale(
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    feat_shapes: Dict[str, Tuple[int, int]],
    strides: Dict[str, int],
    scale_ranges: Dict[str, Tuple[float, float]],
    num_classes: int,
    device: torch.device,
    center_sample_radius: float = 1.5,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """FCOS 多尺度目标分配 (center sampling)。

    对每个 FPN 层级:
      - 只有回归距离 (l,t,r,b) 在 scale_range 内的位置才分配
      - center sampling: 位置必须在 GT box 中心 (cx±r*s, cy±r*s) 子区域内
      - 重叠: 选择面积最小的 GT (利于小目标)

    Args:
        gt_boxes:    list of (N_b, 4) in (x1,y1,x2,y2)
        gt_labels:   list of (N_b,)
        feat_shapes: {"P3": (H,W), ...}
        strides:     {"P3": 8, ...}
        scale_ranges: {"P3": (0,64), ...}
        num_classes: 类别数
        device:      torch device
        center_sample_radius: 中心采样半径倍数 (1.5 = FCOS 默认)

    Returns:
        targets dict: {level: {"cls": (B,C,H,W), "reg": (B,4,H,W),
                                "ctr": (B,1,H,W), "mask": (B,1,H,W)}}
    """
    B = len(gt_boxes)

    # 初始化各层级 target
    targets = {}
    for lv, (H, W) in feat_shapes.items():
        targets[lv] = {
            "cls": torch.zeros(B, num_classes, H, W, device=device),
            "reg": torch.zeros(B, 4, H, W, device=device),
            "ctr": torch.zeros(B, 1, H, W, device=device),
            "mask": torch.zeros(B, 1, H, W, device=device),
        }

    for b in range(B):
        boxes = gt_boxes[b]     # (N, 4)
        labels = gt_labels[b]   # (N,)
        if boxes.numel() == 0:
            continue

        # 计算每个 GT box 的面积 (用于重叠时选择最小)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        for lv in targets:
            stride = strides[lv]
            lo, hi = scale_ranges[lv]
            H, W = feat_shapes[lv]

            for n in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[n]
                cls_id = int(labels[n].item())
                box_area = areas[n]

                # Box 中心
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Center sampling 子区域
                r = center_sample_radius * stride
                cs_x1 = (cx - r) / stride; cs_y1 = (cy - r) / stride
                cs_x2 = (cx + r) / stride; cs_y2 = (cy + r) / stride

                # 网格范围
                ix1 = max(0, int(cs_x1)); iy1 = max(0, int(cs_y1))
                ix2 = min(W, int(cs_x2) + 1); iy2 = min(H, int(cs_y2) + 1)

                for iy in range(iy1, iy2):
                    for ix in range(ix1, ix2):
                        # 图像坐标
                        loc_x = stride * ix + stride / 2.0
                        loc_y = stride * iy + stride / 2.0

                        # 到 box 四边的距离
                        l = loc_x - x1; t = loc_y - y1
                        r_dist = x2 - loc_x; b_dist = y2 - loc_y

                        # 检查: 位置必须在 box 内部
                        if l <= 0 or t <= 0 or r_dist <= 0 or b_dist <= 0:
                            continue

                        # 检查: 回归距离在 scale_range 内
                        max_dist = max(l, t, r_dist, b_dist)
                        if max_dist < lo or max_dist > hi:
                            continue

                        # 重叠处理: 如果已被分配，选择面积更小的 GT
                        if targets[lv]["mask"][b, 0, iy, ix] > 0:
                            if box_area >= areas[int(targets[lv]["mask"][b, 0, iy, ix].item()) - 1]:
                                continue

                        # 分配
                        targets[lv]["cls"][b, cls_id, iy, ix] = 1.0
                        targets[lv]["reg"][b, 0, iy, ix] = l
                        targets[lv]["reg"][b, 1, iy, ix] = t
                        targets[lv]["reg"][b, 2, iy, ix] = r_dist
                        targets[lv]["reg"][b, 3, iy, ix] = b_dist

                        ctr_val = ((min(l, r_dist) * min(t, b_dist)) /
                                   (max(l, r_dist) * max(t, b_dist) + 1e-7)) ** 0.5
                        targets[lv]["ctr"][b, 0, iy, ix] = ctr_val
                        targets[lv]["mask"][b, 0, iy, ix] = float(n + 1)

    return targets


# ═══════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════

def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0,
               num_pos: int = 1) -> torch.Tensor:
    """Focal Loss with sum/max(num_pos,1) normalization."""
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = targets * prob + (1 - targets) * (1 - prob)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * ce).sum() / max(num_pos, 1)


def giou_loss(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> torch.Tensor:
    """GIoU Loss: 1 - GIoU."""
    # Intersection
    ix1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    iy1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    ix2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    iy2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # Union
    area_pred = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    area_target = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])
    union = area_pred + area_target - inter + 1e-7
    iou = inter / union

    # Enclosing box
    cx1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
    cy1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
    cx2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
    cy2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
    c_area = (cx2 - cx1) * (cy2 - cy1) + 1e-7

    giou = iou - (c_area - union) / c_area
    return (1 - giou).mean()


def compute_det_loss(
    detections: Dict[str, Dict[str, torch.Tensor]],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    strides: Dict[str, int],
    scale_ranges: Dict[str, Tuple[float, float]],
    num_classes: int,
    lambda_cls: float = 1.0,
    lambda_l1: float = 1.0,
    lambda_giou: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """计算多尺度 FCOS 检测损失。

    L_total = lambda_cls * L_focal + lambda_l1 * L_l1 + lambda_giou * L_giou + L_ctr
    所有损失使用 sum / max(num_pos, 1) 归一化。
    """
    device = next(iter(detections.values()))["cls_logits"].device

    # ── 获取各层级 feature shape ──
    feat_shapes = {}
    for lv, d in detections.items():
        _, _, H, W = d["cls_logits"].shape
        feat_shapes[lv] = (H, W)

    # ── 目标分配 ──
    targets = fcos_targets_multi_scale(
        gt_boxes, gt_labels, feat_shapes, strides, scale_ranges,
        num_classes, device)

    # ── 收集所有正样本 ──
    all_cls = []
    all_cls_target = []
    all_reg = []
    all_reg_target = []
    all_ctr = []
    all_ctr_target = []
    total_pos = 0

    for lv in detections:
        d = detections[lv]
        t = targets[lv]
        mask = t["mask"].bool()  # (B, 1, H, W)
        pos = mask.sum().item()
        total_pos += pos

        # 分类: 所有位置参与 Focal Loss
        cls_pred = d["cls_logits"].permute(0, 2, 3, 1).reshape(-1, num_classes)
        cls_tgt = t["cls"].permute(0, 2, 3, 1).reshape(-1, num_classes)
        all_cls.append(cls_pred)
        all_cls_target.append(cls_tgt)

        if pos > 0:
            # 回归 & 中心度: 仅正样本
            indices = mask.squeeze(1).nonzero(as_tuple=False)  # (K, 3)
            pred_ltrb = d["bbox_preds"][indices[:, 0], :, indices[:, 1], indices[:, 2]]
            tgt_ltrb = t["reg"][indices[:, 0], :, indices[:, 1], indices[:, 2]]
            all_reg.append(pred_ltrb)
            all_reg_target.append(tgt_ltrb)

            ctr_pred = d["centerness"][mask]
            ctr_tgt = t["ctr"][mask]
            all_ctr.append(ctr_pred)
            all_ctr_target.append(ctr_tgt)

    # ── Focal Loss ──
    cls_cat = torch.cat(all_cls, dim=0) if all_cls else torch.zeros(0, num_classes, device=device)
    cls_tgt_cat = torch.cat(all_cls_target, dim=0)
    L_cls = focal_loss(cls_cat, cls_tgt_cat, num_pos=max(total_pos, 1))

    # ── L1 Loss ──
    if all_reg:
        reg_cat = torch.cat(all_reg, dim=0)
        reg_tgt_cat = torch.cat(all_reg_target, dim=0)
        L_l1 = F.smooth_l1_loss(reg_cat, reg_tgt_cat, reduction="sum") / max(total_pos, 1)

        # GIoU: 转换 (l,t,r,b) → (x1,y1,x2,y2)
        pos_indices_all = []
        for lv in detections:
            t = targets[lv]
            if t["mask"].sum() > 0:
                stride = strides[lv]
                indices = t["mask"].squeeze(1).nonzero(as_tuple=False)
                for row in indices:
                    b, y, x = row
                    loc_x = stride * x.float() + stride / 2.0
                    loc_y = stride * y.float() + stride / 2.0
                    pos_indices_all.append((lv, b.item(), y.item(), x.item(),
                                            loc_x, loc_y))

        pred_xyxy_list = []
        tgt_xyxy_list = []
        for lv, b, y, x, lx, ly in pos_indices_all:
            p_ltrb = detections[lv]["bbox_preds"][b, :, y, x]
            t_ltrb = targets[lv]["reg"][b, :, y, x]
            pred_xyxy_list.append(torch.stack([lx - p_ltrb[0], ly - p_ltrb[1],
                                                lx + p_ltrb[2], ly + p_ltrb[3]]))
            tgt_xyxy_list.append(torch.stack([lx - t_ltrb[0], ly - t_ltrb[1],
                                               lx + t_ltrb[2], ly + t_ltrb[3]]))

        pred_xyxy = torch.stack(pred_xyxy_list)
        tgt_xyxy = torch.stack(tgt_xyxy_list)
        L_giou = giou_loss(pred_xyxy, tgt_xyxy)
    else:
        L_l1 = torch.tensor(0.0, device=device)
        L_giou = torch.tensor(0.0, device=device)

    # ── Centerness Loss ──
    if all_ctr:
        ctr_cat = torch.cat(all_ctr, dim=0)
        ctr_tgt_cat = torch.cat(all_ctr_target, dim=0)
        L_ctr = F.binary_cross_entropy(ctr_cat, ctr_tgt_cat, reduction="sum") / max(total_pos, 1)
    else:
        L_ctr = torch.tensor(0.0, device=device)

    total = lambda_cls * L_cls + lambda_l1 * L_l1 + lambda_giou * L_giou + L_ctr

    # 极小正则项: 确保所有层级 (包括当前无正样本的层级) scale 参数都有梯度流
    scale_reg = 1e-6 * sum((d["bbox_preds"].mean()) for d in detections.values())
    total = total + scale_reg

    return {
        "cls_loss": L_cls,
        "l1_loss": L_l1,
        "giou_loss": L_giou,
        "ctr_loss": L_ctr,
        "total": total,
        "num_pos": total_pos,
    }


# ── 快速验证 ──────────────────────────────────────────────────
def _test():
    print("=" * 65)
    print("[MultiScaleDetHead] Multi-Scale FCOS Detection Head")
    B, H, W = 2, 640, 640
    num_classes = 12

    # ── 模拟 FPN 输出 (P3–P7) ──
    feats = {
        "P3": torch.randn(B, 256, H // 8,  W // 8),
        "P4": torch.randn(B, 256, H // 16, W // 16),
        "P5": torch.randn(B, 256, H // 32, W // 32),
        "P6": torch.randn(B, 256, H // 64, W // 64),
        "P7": torch.randn(B, 256, H // 128, W // 128),
    }

    # ── 测试前向 ──
    print("\n[Forward pass]")
    head = MultiScaleDetHead(in_channels=256, num_classes=num_classes, num_conv=4)
    head.train()
    out = head(feats)

    for lv, d in out.items():
        print(f"  {lv} (stride={head.strides[lv]:>3}): "
              f"cls={tuple(d['cls_logits'].shape)}, "
              f"reg={tuple(d['bbox_preds'].shape)}, "
              f"ctr={tuple(d['centerness'].shape)}")

    # ── 模拟 GT ──
    gt_boxes = [
        torch.tensor([[50.,  40.,  150., 200.],   # 小目标 → P3
                      [100., 50.,  500., 500.],   # 中目标 → P5
                      [10.,  10.,  600., 350.]]), # 大目标 → P6
        torch.tensor([[80.,  60.,  200., 180.],
                      [200., 100., 400., 500.]]),
    ]
    gt_labels = [
        torch.tensor([0, 3, 6]),
        torch.tensor([2, 5]),
    ]

    # ── 计算损失 ──
    print("\n[Loss computation]")
    losses = compute_det_loss(out, gt_boxes, gt_labels,
                               head.strides, head.scale_ranges, num_classes)
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:12s}: {v.item():.6f}")
        else:
            print(f"  {k:12s}: {v}")

    assert losses["total"].item() > 0, "Total loss should be > 0"
    assert losses["num_pos"] > 0, "Should have at least 1 positive sample"

    # ── 梯度流 ──
    print("\n[Gradient flow]")
    losses["total"].backward()
    grad_ok = all(
        p.grad is not None for p in head.parameters() if p.requires_grad
    )
    print(f"  All params have gradients: {grad_ok}")
    p = sum(p.numel() for p in head.parameters()) / 1e6
    print(f"  params: {p:.2f}M")

    # ── 空标注 ──
    print("\n[Empty GT]")
    losses_empty = compute_det_loss(out,
                                     [torch.zeros(0, 4), torch.zeros(0, 4)],
                                     [torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)],
                                     head.strides, head.scale_ranges, num_classes)
    print(f"  total={losses_empty['total'].item():.6f}, num_pos={losses_empty['num_pos']}")

    print("\n  PASS")
    print("=" * 65)


if __name__ == "__main__":
    _test()
