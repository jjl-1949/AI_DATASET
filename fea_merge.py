"""
RGB-DT DCR-CBAM multi-scale feature fusion.

The repository performs fusion after the two FPN branches:

    RGB P2-P5  ----\
                    DCR-CBAM -> FP2-FP5
    DT  P2-P5  ----/

Every RGB/DT FPN feature has 256 channels. Features at the same
pyramid level are concatenated to 512 channels and dynamically
reduced back to 256 channels before spatial attention.

Two dynamic reduction modes are provided:

    "full":
        Paper-style full dynamic matrix W_red(Mc). This is the closest
        implementation of equations (7)-(9), but one 512->256 block
        contains about 67M generator parameters.

    "low_rank":
        Practical low-rank dynamic matrix
        W_red(Mc) = W_base + scale * A(Mc)B(Mc).
        This keeps input-conditioned reduction while using much less
        memory. It is the recommended default for this repository.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn


TensorDict = Dict[str, torch.Tensor]
AttentionDict = Dict[str, Dict[str, torch.Tensor]]


class ChannelAttention(nn.Module):
    """CBAM channel attention: what feature channels are important."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = max(channels // reduction, 16)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        avg_descriptor = torch.mean(feature, dim=(2, 3), keepdim=True)
        max_descriptor = torch.amax(feature, dim=(2, 3), keepdim=True)
        logits = (
            self.shared_mlp(avg_descriptor)
            + self.shared_mlp(max_descriptor)
        )
        return torch.sigmoid(logits)


class SpatialAttention(nn.Module):
    """CBAM spatial attention: where important features are located."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Spatial-attention kernel_size must be odd.")

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(feature, dim=1, keepdim=True)
        max_map = torch.amax(feature, dim=1, keepdim=True)
        pooled = torch.cat([avg_map, max_map], dim=1)
        return torch.sigmoid(self.conv(pooled))


class FullDynamicReduction(nn.Module):
    """Paper-style full dynamic channel reduction.

    Mc is projected directly to all Cout*Cin elements of W_red:

        W_red = reshape(sigmoid(Linear(Mc)), Cout, Cin)
        F_red = W_red @ F

    This is faithful to the paper equations but parameter-heavy.
    """

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.weight_generator = nn.Linear(
            input_channels,
            output_channels * input_channels,
        )

    def forward(
        self,
        feature: torch.Tensor,
        channel_descriptor: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = feature.shape
        if channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got {channels}."
            )

        dynamic_weight = torch.sigmoid(
            self.weight_generator(channel_descriptor)
        ).view(batch_size, self.output_channels, self.input_channels)

        flattened = feature.flatten(2)
        reduced = torch.bmm(dynamic_weight, flattened)
        return reduced.view(
            batch_size,
            self.output_channels,
            height,
            width,
        )


class LowRankDynamicReduction(nn.Module):
    """Memory-efficient input-conditioned channel reduction.

    A low-rank update is generated for every sample:

        W_red(Mc) = W_base + scale * tanh(A(Mc) @ B(Mc))
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        rank: int = 8,
        hidden_channels: int = 64,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.rank = rank

        self.base_weight = nn.Parameter(
            torch.empty(output_channels, input_channels)
        )
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)

        self.condition_encoder = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.a_generator = nn.Linear(
            hidden_channels,
            output_channels * rank,
        )
        self.b_generator = nn.Linear(
            hidden_channels,
            rank * input_channels,
        )

        # Start close to a conventional static 1x1 reduction and gradually
        # learn the sample-specific update.
        self.dynamic_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        feature: torch.Tensor,
        channel_descriptor: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = feature.shape
        if channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got {channels}."
            )

        condition = self.condition_encoder(channel_descriptor)
        matrix_a = self.a_generator(condition).view(
            batch_size,
            self.output_channels,
            self.rank,
        )
        matrix_b = self.b_generator(condition).view(
            batch_size,
            self.rank,
            self.input_channels,
        )

        dynamic_delta = torch.bmm(matrix_a, matrix_b) / (self.rank ** 0.5)
        dynamic_weight = (
            self.base_weight.unsqueeze(0)
            + self.dynamic_scale * torch.tanh(dynamic_delta)
        )

        flattened = feature.flatten(2)
        reduced = torch.bmm(dynamic_weight, flattened)
        return reduced.view(
            batch_size,
            self.output_channels,
            height,
            width,
        )


class DCRCBAMBlock(nn.Module):
    """DCR-CBAM block for one matching RGB/DT FPN level."""

    def __init__(
        self,
        rgb_channels: int = 256,
        dt_channels: int = 256,
        output_channels: int = 256,
        channel_reduction: int = 16,
        spatial_kernel_size: int = 7,
        reduction_mode: str = "low_rank",
        dynamic_rank: int = 8,
    ):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.dt_channels = dt_channels
        self.input_channels = rgb_channels + dt_channels
        self.output_channels = output_channels

        self.channel_attention = ChannelAttention(
            channels=self.input_channels,
            reduction=channel_reduction,
        )

        if reduction_mode == "full":
            self.dynamic_reduction = FullDynamicReduction(
                input_channels=self.input_channels,
                output_channels=output_channels,
            )
        elif reduction_mode == "low_rank":
            self.dynamic_reduction = LowRankDynamicReduction(
                input_channels=self.input_channels,
                output_channels=output_channels,
                rank=dynamic_rank,
            )
        else:
            raise ValueError(
                "reduction_mode must be either 'full' or 'low_rank'."
            )

        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel_size
        )

    def _validate_inputs(
        self,
        rgb_feature: torch.Tensor,
        dt_feature: torch.Tensor,
    ) -> None:
        if rgb_feature.ndim != 4 or dt_feature.ndim != 4:
            raise ValueError("RGB and DT features must be 4D NCHW tensors.")
        if rgb_feature.shape[0] != dt_feature.shape[0]:
            raise ValueError("RGB and DT batch sizes do not match.")
        if rgb_feature.shape[2:] != dt_feature.shape[2:]:
            raise ValueError("RGB and DT spatial sizes do not match.")
        if rgb_feature.shape[1] != self.rgb_channels:
            raise ValueError(
                f"Expected {self.rgb_channels} RGB channels, "
                f"got {rgb_feature.shape[1]}."
            )
        if dt_feature.shape[1] != self.dt_channels:
            raise ValueError(
                f"Expected {self.dt_channels} DT channels, "
                f"got {dt_feature.shape[1]}."
            )

    def forward(
        self,
        rgb_feature: torch.Tensor,
        dt_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._validate_inputs(rgb_feature, dt_feature)

        # F = [P_i^RGB ; P_i^DT], shape: B x 512 x H_i x W_i
        feature = torch.cat([rgb_feature, dt_feature], dim=1)

        # Equations (3) and (5): F' = Mc(F) * F
        channel_map = self.channel_attention(feature)
        channel_refined = channel_map * feature

        # Equations (7)-(9): dynamically reduce 512 channels to 256.
        channel_descriptor = channel_map.flatten(1)
        reduced = self.dynamic_reduction(
            channel_refined,
            channel_descriptor,
        )

        # Equations (4) and (6): F''_red = Ms(F'_red) * F'_red
        spatial_map = self.spatial_attention(reduced)
        fused = spatial_map * reduced

        attention = {
            "channel": channel_map,
            "spatial": spatial_map,
        }
        return fused, attention


class MultiScaleDCRCBAM(nn.Module):
    """Apply independent DCR-CBAM blocks to P2, P3, P4 and P5."""

    DEFAULT_LEVELS = ("P2", "P3", "P4", "P5")

    def __init__(
        self,
        channels: int = 256,
        levels: Tuple[str, ...] = DEFAULT_LEVELS,
        channel_reduction: int = 16,
        spatial_kernel_size: int = 7,
        reduction_mode: str = "low_rank",
        dynamic_rank: int = 8,
    ):
        super().__init__()
        self.levels = levels
        self.blocks = nn.ModuleDict({
            level: DCRCBAMBlock(
                rgb_channels=channels,
                dt_channels=channels,
                output_channels=channels,
                channel_reduction=channel_reduction,
                spatial_kernel_size=spatial_kernel_size,
                reduction_mode=reduction_mode,
                dynamic_rank=dynamic_rank,
            )
            for level in levels
        })

    def forward(
        self,
        rgb_features: Mapping[str, torch.Tensor],
        dt_features: Mapping[str, torch.Tensor],
    ) -> Tuple[TensorDict, AttentionDict]:
        fused_features: TensorDict = {}
        attention_maps: AttentionDict = {}

        for level in self.levels:
            if level not in rgb_features:
                raise KeyError(f"RGB feature dictionary is missing {level}.")
            if level not in dt_features:
                raise KeyError(f"DT feature dictionary is missing {level}.")

            fused, attention = self.blocks[level](
                rgb_features[level],
                dt_features[level],
            )
            fused_features[level] = fused
            attention_maps[level] = attention

        return fused_features, attention_maps


def channel_saliency_loss(
    attention_maps: Mapping[str, Mapping[str, torch.Tensor]],
) -> torch.Tensor:
    """Paper channel-saliency regularizer: sum of mean |Mc| per level."""
    losses = [
        maps["channel"].abs().mean()
        for maps in attention_maps.values()
    ]
    if not losses:
        raise ValueError("attention_maps must contain at least one level.")
    return torch.stack(losses).sum()


def _test() -> None:
    torch.manual_seed(7)
    batch_size, height, width = 2, 128, 128
    sizes = {
        "P2": (height // 4, width // 4),
        "P3": (height // 8, width // 8),
        "P4": (height // 16, width // 16),
        "P5": (height // 32, width // 32),
    }
    rgb_features = {
        level: torch.randn(batch_size, 256, h, w, requires_grad=True)
        for level, (h, w) in sizes.items()
    }
    dt_features = {
        level: torch.randn(batch_size, 256, h, w, requires_grad=True)
        for level, (h, w) in sizes.items()
    }

    model = MultiScaleDCRCBAM(
        channels=256,
        reduction_mode="low_rank",
        dynamic_rank=8,
    )
    fused_features, attention_maps = model(rgb_features, dt_features)

    total_loss = sum(x.mean() for x in fused_features.values())
    total_loss = total_loss + 1e-4 * channel_saliency_loss(attention_maps)
    total_loss.backward()

    for level in model.levels:
        expected_shape = rgb_features[level].shape
        assert fused_features[level].shape == expected_shape
        assert attention_maps[level]["channel"].shape == (
            batch_size, 512, 1, 1
        )
        assert attention_maps[level]["spatial"].shape == (
            batch_size, 1, *sizes[level]
        )
        assert rgb_features[level].grad is not None
        assert dt_features[level].grad is not None
        print(
            f"{level}: fused={tuple(fused_features[level].shape)}, "
            f"Mc={tuple(attention_maps[level]['channel'].shape)}, "
            f"Ms={tuple(attention_maps[level]['spatial'].shape)}"
        )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={parameters / 1e6:.2f}M")
    print("DCR-CBAM forward/backward test: PASS")


if __name__ == "__main__":
    _test()
