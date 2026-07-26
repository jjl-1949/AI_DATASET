"""
ResNet50 Backbone — 共享骨干网络。

支持多输入通道 / 预训练 & 从头训练。
输出多尺度特征图 {"C2","C3","C4","C5"}，供后续 FPN 使用。

参考: Deep Residual Learning for Image Recognition (He et al., 2016)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class Bottleneck(nn.Module):
    """ResNet Bottleneck block: 1×1 → 3×3 → 1×1, expansion=4."""

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50Backbone(nn.Module):
    """ResNet50 骨干网络。

    参数:
        in_channels: 输入通道数 (RGB=3, IR=1, Depth=1).
        pretrained: 是否加载 ImageNet 预训练权重。仅 in_channels=3 时有效。
        frozen_stages: 冻结前几个 stage (-1=不冻结, 1=stem+layer1, 2=...+layer2, ...).

    输出:
        Dict[str, Tensor]  键 'C2','C3','C4','C5'，stride 4/8/16/32.
    """

    LAYERS = [3, 4, 6, 3]
    PLANES = [64, 128, 256, 512]

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = False,
        frozen_stages: int = -1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.frozen_stages = frozen_stages

        # ── Stem ──────────────────────────────────────────
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── Layers 1-4 ────────────────────────────────────
        self.inplanes = 64
        self.layer1 = self._make_layer(64, self.LAYERS[0], stride=1)   # /4,  256c
        self.layer2 = self._make_layer(128, self.LAYERS[1], stride=2)  # /8,  512c
        self.layer3 = self._make_layer(256, self.LAYERS[2], stride=2)  # /16, 1024c
        self.layer4 = self._make_layer(512, self.LAYERS[3], stride=2)  # /32, 2048c

        # ── 初始化 ─────────────────────────────────────────
        if pretrained and in_channels == 3:
            self._load_pretrained()
        else:
            self._init_weights()

        if frozen_stages >= 0:
            self._freeze_stages(frozen_stages)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * Bottleneck.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    # ── 预训练加载 ────────────────────────────────────────
    def _load_pretrained(self):
        import torchvision.models as tv_models
        pretrained_resnet = tv_models.resnet50(weights="IMAGENET1K_V2")
        state_dict = {
            k: v for k, v in pretrained_resnet.state_dict().items()
            if not k.startswith("fc.")
        }
        if self.in_channels != 3:
            del state_dict["conv1.weight"]
        self.load_state_dict(state_dict, strict=False)

    # ── Kaiming 初始化 ────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu",
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ── 冻结 ──────────────────────────────────────────────
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
        for module in freeze_list:
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_stages >= 0:
            self._freeze_stages(self.frozen_stages)
        return self

    # ── 前向 ──────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)       # stride 4

        C2 = self.layer1(x)       # stride 4,  256c
        C3 = self.layer2(C2)      # stride 8,  512c
        C4 = self.layer3(C3)      # stride 16, 1024c
        C5 = self.layer4(C4)      # stride 32, 2048c

        return {"C2": C2, "C3": C3, "C4": C4, "C5": C5}
