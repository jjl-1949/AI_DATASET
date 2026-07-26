"""
深度 (Depth) 特征提取模块 — ResNet50 从头训练。

输入:  (B, 1, H, W)  深度图 (单通道, 值域建议归一化到 [0,1] 或标准化)
输出:  {"C2": (B,256,H/4,W/4),
        "C3": (B,512,H/8,W/8),
        "C4": (B,1024,H/16,W/16),
        "C5": (B,2048,H/32,W/32)}

策略: 深度图的像素值含义与 RGB/红外不同 (距离 vs 辐射强度)，
      不建议复用预训练权重，应从头学习深度特有的几何结构特征。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════
# Bottleneck
# ═══════════════════════════════════════════════════════════════
class Bottleneck(nn.Module):
    """ResNet Bottleneck: 1×1 → 3×3 → 1×1, expansion=4."""

    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# ═══════════════════════════════════════════════════════════════
# ResNet50 骨干
# ═══════════════════════════════════════════════════════════════
class ResNet50Backbone(nn.Module):
    """ResNet50 骨干，输出多尺度特征 C2-C5。"""

    LAYERS = [3, 4, 6, 3]
    PLANES = [64, 128, 256, 512]

    def __init__(self, in_channels: int = 1, pretrained: bool = False,
                 frozen_stages: int = -1):
        super().__init__()
        self.frozen_stages = frozen_stages

        # Stem — conv1 输入通道 = 1 (深度单通道)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers
        self.inplanes = 64
        self.layer1 = self._make_layer(64,  self.LAYERS[0], stride=1)  # /4,  256c
        self.layer2 = self._make_layer(128, self.LAYERS[1], stride=2)  # /8,  512c
        self.layer3 = self._make_layer(256, self.LAYERS[2], stride=2)  # /16, 1024c
        self.layer4 = self._make_layer(512, self.LAYERS[3], stride=2)  # /32, 2048c

        if pretrained:
            self._load_pretrained()
        else:
            self._init_weights()

        if frozen_stages >= 0:
            self._freeze_stages(frozen_stages)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _load_pretrained(self):
        import torchvision.models as tv_models
        state_dict = {k: v for k, v in
                      tv_models.resnet50(weights="IMAGENET1K_V2").state_dict().items()
                      if not k.startswith("fc.")}
        del state_dict["conv1.weight"]
        self.load_state_dict(state_dict, strict=False)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze_stages(self, stage: int):
        freeze_list = []
        if stage >= 1:
            freeze_list.extend([self.conv1, self.bn1, self.layer1])
        if stage >= 2:
            freeze_list.append(self.layer2)
        if stage >= 3:
            freeze_list.append(self.layer3)
        if stage >= 4:
            freeze_list.append(self.layer4)
        for mod in freeze_list:
            for p in mod.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_stages >= 0:
            self._freeze_stages(self.frozen_stages)
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        C2 = self.layer1(x)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)
        return {"C2": C2, "C3": C3, "C4": C4, "C5": C5}


# ═══════════════════════════════════════════════════════════════
# 深度分支
# ═══════════════════════════════════════════════════════════════
class DepthBranch(nn.Module):
    """深度分支 — ResNet50 从头训练。"""

    def __init__(self, frozen_stages: int = -1):
        super().__init__()
        self.backbone = ResNet50Backbone(
            in_channels=1, pretrained=False, frozen_stages=frozen_stages)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.backbone(x)


# ── 快速验证 ──────────────────────────────────────────────────
def _test():
    print("=" * 50)
    print("[DepthBranch] ResNet50 from scratch")
    B, H, W = 2, 640, 640
    model = DepthBranch()
    model.train()
    x = torch.randn(B, 1, H, W)
    feats = model(x)
    for k, v in feats.items():
        print(f"  {k}: {tuple(v.shape)}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
    print("  PASS")
    print("=" * 50)


if __name__ == "__main__":
    _test()
