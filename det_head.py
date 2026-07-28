"""
Detection Head — FCOS 风格 Anchor-Free 检测头

在 DMLab 输出的高分辨率特征图上 (stride=4)，每个空间位置直接预测:
  - 类别分数 (12 类)
  - 边界框距离 (l, t, r, b)
  - 中心度 (centerness): 预测质量分数，抑制偏离目标中心的低质量预测

FCOS 回归格式:
  对于特征图上位置 (i, j), 对应原图坐标 (s·i + s/2, s·j + s/2), s = stride
  预测 (l, t, r, b) 分别表示该点到目标框左/上/右/下边界的距离 (像素单位)

  从 (l,t,r,b) 转换到 YOLO (cx,cy,w,h):
    w = l + r                       (框宽)
    h = t + b                       (框高)
    cx = loc_x - l + w/2           (中心 x)
    cy = loc_y - t + h/2           (中心 y)

输入:  (B, 256, H/4, W/4)           ← 来自 DMLab 的输出

输出:  {
    "cls_logits":  (B, 12, H/4, W/4),   # 类别 logits (不含背景)
    "bbox_preds":  (B, 4,  H/4, W/4),   # 框距离 (l, t, r, b) > 0
    "centerness":  (B, 1,  H/4, W/4),   # 中心度 ∈ [0, 1]
}

后续: 输出用于 Loss 计算 (Focal Loss + GIoU Loss + BCE)，训练时 centerness
      与分类分数相乘作为最终置信度。
"""

import torch
import torch.nn as nn
from typing import Dict


# ═══════════════════════════════════════════════════════════════
# Conv-BN-ReLU 构建块
# ═══════════════════════════════════════════════════════════════
def _make_conv_block(in_channels: int, out_channels: int,
                     num_convs: int) -> nn.Sequential:
    """构建 num_convs 层 Conv3×3 → BN → ReLU 堆叠。"""
    layers = []
    for _ in range(num_convs):
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
        in_channels = out_channels  # 后续层 in=out
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════
# Detection Head
# ═══════════════════════════════════════════════════════════════
class DetHead(nn.Module):
    """FCOS 风格 Anchor-Free 检测头。

    三个独立子网络:
      - 分类子网: num_conv × Conv3×3 + 输出 Conv → 类别 logits
      - 回归子网: num_conv × Conv3×3 + 输出 Conv → 框距离 (l,t,r,b)
      - 中心度子网: num_conv × Conv3×3 + 输出 Conv → centerness

    所有子网络独立学习，不共享参数，各司其职。

    Args:
        in_channels: 输入通道数 (默认 256, 与 FPN/DMLab 输出一致)
        num_classes: 类别数 (默认 12)
        num_conv:    每个子网中的卷积层数 (默认 4, FCOS 标准)
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 12,
                 num_conv: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # ── 分类子网 ──
        self.cls_convs = _make_conv_block(in_channels, in_channels, num_conv)
        self.cls_out = nn.Conv2d(in_channels, num_classes, kernel_size=3,
                                 padding=1)

        # ── 回归子网 ──
        self.reg_convs = _make_conv_block(in_channels, in_channels, num_conv)
        self.reg_out = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)

        # ── 中心度子网 ──
        self.ctr_convs = _make_conv_block(in_channels, in_channels, num_conv)
        self.ctr_out = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        # Stem convs: Kaiming init — standard for ReLU-activated conv layers
        # Output convs: small normal init — prevents sigmoid saturation and
        #   ensures stable initial predictions (FCOS standard).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                is_output = m in (self.cls_out, self.reg_out, self.ctr_out)
                if is_output:
                    # Small init: prevent logit explosion in output layers
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                if m.bias is not None:
                    if m is self.cls_out:
                        # prior_prob=0.01 → initial output ≈ sigmoid(-4.595) ≈ 0.01
                        prior_prob = 0.01
                        nn.init.constant_(m.bias, -torch.log(
                            torch.tensor((1 - prior_prob) / prior_prob)))
                    elif m is self.reg_out:
                        # bias=0 → exp(0)=1 → initial bbox ~1 pixel from center
                        nn.init.constant_(m.bias, 0.0)
                    elif m is self.ctr_out:
                        # bias=0 → sigmoid(0)=0.5 → initial centerness ~0.5
                        nn.init.constant_(m.bias, 0.0)
                    else:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播。

        Args:
            x: DMLab 输出特征 (B, in_channels, H/4, W/4)

        Returns:
            字典包含 cls_logits, bbox_preds, centerness
        """
        # 1. 分类: (B, C, H/4, W/4) → (B, num_classes, H/4, W/4)
        cls_logits = self.cls_out(self.cls_convs(x))

        # 2. 回归: (B, C, H/4, W/4) → (B, 4, H/4, W/4)
        #    exp() ensures (l, t, r, b) > 0 with non-zero gradient everywhere
        bbox_preds = torch.exp(self.reg_out(self.reg_convs(x)))

        # 3. 中心度: (B, C, H/4, W/4) → (B, 1, H/4, W/4)
        #    Sigmoid 约束到 [0, 1]
        centerness = torch.sigmoid(self.ctr_out(self.ctr_convs(x)))

        return {
            "cls_logits": cls_logits,
            "bbox_preds": bbox_preds,
            "centerness": centerness,
        }


# ── 快速验证 ──────────────────────────────────────────────────
def _test():
    print("=" * 60)
    print("[DetHead] FCOS-style Anchor-Free Detection Head")

    B, C, H, W = 2, 256, 160, 160  # stride-4 for 640×640 input
    num_classes = 12

    # ── 模拟 DMLab 输出 ──
    x = torch.randn(B, C, H, W)

    # ── 测试默认配置 ──
    print("\n[Default config: num_classes=12, num_conv=4]")
    head = DetHead(in_channels=256, num_classes=num_classes, num_conv=4)
    head.train()

    out = head(x)

    expected = {
        "cls_logits": (B, num_classes, H, W),
        "bbox_preds": (B, 4, H, W),
        "centerness": (B, 1, H, W),
    }
    for k, exp_shape in expected.items():
        assert out[k].shape == exp_shape, \
            f"{k}: {tuple(out[k].shape)} != {exp_shape}"
        print(f"  {k}: {tuple(out[k].shape)}  OK")

    # ── 验证输出值域 ──
    print("\n[Value range check]")
    head.eval()
    with torch.no_grad():
        out2 = head(x)
        print(f"  cls_logits range: [{out2['cls_logits'].min():.3f}, "
              f"{out2['cls_logits'].max():.3f}]")
        print(f"  bbox_preds min:   {out2['bbox_preds'].min():.6f}  (should be > 0)")
        print(f"  centerness range: [{out2['centerness'].min():.4f}, "
              f"{out2['centerness'].max():.4f}]  (should be in [0,1])")

        assert out2["bbox_preds"].min() > 0, "bbox_preds should be > 0 (exp())"
        assert 0 <= out2["centerness"].min() <= out2["centerness"].max() <= 1, \
            "centerness should be in [0, 1]"

    # ── 测试非正方形输入 ──
    print("\n[Non-square input 360×640 → stride-4: 90×160]")
    x2 = torch.randn(B, C, 90, 160)
    out3 = head(x2)
    for k, v in out3.items():
        print(f"  {k}: {tuple(v.shape)}")
    assert out3["cls_logits"].shape == (B, num_classes, 90, 160)
    assert out3["bbox_preds"].shape == (B, 4, 90, 160)
    assert out3["centerness"].shape == (B, 1, 90, 160)
    print("  OK")

    # ── 测试不同 num_conv 配置 ──
    print("\n[Config: num_conv=2]")
    head2 = DetHead(in_channels=256, num_classes=num_classes, num_conv=2)
    out4 = head2(x)
    print(f"  cls_logits: {tuple(out4['cls_logits'].shape)}  OK")

    # ── 参数量 ──
    total = sum(p.numel() for p in head.parameters())
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\n  params: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
    print("  PASS")
    print("=" * 60)


if __name__ == "__main__":
    _test()
