"""Tests for the ResNet Bottleneck block used across dep_deal, hw_deal, rgb_deal."""

import pytest
import torch
from dep_deal import Bottleneck as BottleneckDepth
from hw_deal import Bottleneck as BottleneckIR
from rgb_deal import Bottleneck as BottleneckRGB


class TestBottleneck:
    """Test all three Bottleneck implementations (they are identical copies)."""

    IMPLS = [BottleneckDepth, BottleneckIR, BottleneckRGB]

    # ── Basic shape tests ───────────────────────────────────────

    @pytest.mark.parametrize("bottleneck_cls", IMPLS)
    def test_output_shape_stride1(self, bottleneck_cls, device):
        """Output spatial dim = input, channels = 4×planes (stride=1, matching ch)."""
        B, planes, H, W = 2, 64, 40, 40
        in_c = planes * 4  # 256 in, 256 out
        x = torch.randn(B, in_c, H, W, device=device)

        blk = bottleneck_cls(in_c, planes, stride=1).to(device)
        out = blk(x)

        expected_c = planes * blk.expansion  # 4×planes
        assert out.shape == (B, expected_c, H, W), \
            f"Expected {(B, expected_c, H, W)}, got {tuple(out.shape)}"

    @pytest.mark.parametrize("bottleneck_cls", IMPLS)
    def test_output_shape_stride2(self, bottleneck_cls, device):
        """Stride=2: output spatial dim halved, channels = 4×planes (via backbone)."""
        import torch.nn as nn
        B, planes, H, W = 2, 64, 80, 80
        in_c = planes * 4  # 256
        x = torch.randn(B, in_c, H, W, device=device)

        # Bottleneck needs explicit downsample when stride=2
        downsample = nn.Sequential(
            nn.Conv2d(in_c, planes * 4, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(planes * 4),
        ).to(device)
        blk = bottleneck_cls(in_c, planes, stride=2, downsample=downsample).to(device)
        out = blk(x)

        expected_c = planes * blk.expansion  # 4×planes
        assert out.shape == (B, expected_c, H // 2, W // 2), \
            f"Expected {(B, expected_c, H//2, W//2)}, got {tuple(out.shape)}"

    # ── Downsample behaviour ───────────────────────────────────

    def test_downsample_is_explicitly_provided(self):
        """Bottleneck.downsample is set from the constructor argument."""
        import torch.nn as nn

        # With explicit downsample
        ds = nn.Sequential(nn.Conv2d(256, 512, 1, 2, bias=False))
        blk = BottleneckDepth(256, 128, stride=2, downsample=ds)
        assert blk.downsample is ds

        # Without downsample → None
        blk2 = BottleneckDepth(64, 16, stride=1)
        assert blk2.downsample is None

    def test_backbone_make_layer_creates_downsample(self, device):
        """Backbone._make_layer auto-creates downsample when stride≠1 or ch mismatch."""
        from dep_deal import ResNet50Backbone
        bb = ResNet50Backbone(in_channels=1, frozen_stages=0).to(device)

        # layer2 uses stride=2, inplanes=256→512 → must have downsample
        assert bb.layer2[0].downsample is not None, \
            "First block of layer2 (stride=2) must have downsample"

        # layer1 first block: inplanes=64, out=64*4=256, 64≠256 → downsample created
        assert bb.layer1[0].downsample is not None, \
            "First block of layer1: inplanes=64, out=256 → downsample needed"

    # ── Identity / residual path ──────────────────────────────

    def test_residual_connection(self, device):
        """With stride=1 and matching channels, output ≠ all-zero (residual is active)."""
        B, planes, H, W = 2, 64, 40, 40
        in_c = planes * 4  # 256

        x = torch.randn(B, in_c, H, W, device=device)
        blk = BottleneckDepth(in_c, planes, stride=1).to(device)

        out = blk(x)
        # Residual connection means output should not be trivially all-zero
        assert not torch.allclose(out, torch.zeros_like(out)), "Output should not be all-zero"
        # And should differ from identity in general
        assert not torch.equal(out, x), "With random weights, out ≠ identity"

    # ── Gradient flow ─────────────────────────────────────────

    @pytest.mark.parametrize("bottleneck_cls", IMPLS)
    def test_gradient_flow(self, bottleneck_cls, device):
        """Gradients should flow through all trainable parameters."""
        B, planes, H, W = 2, 64, 40, 40
        in_c = planes * 4

        x = torch.randn(B, in_c, H, W, device=device, requires_grad=False)
        blk = bottleneck_cls(in_c, planes, stride=1).to(device)

        out = blk(x)
        loss = out.sum()
        loss.backward()

        for name, p in blk.named_parameters():
            assert p.grad is not None, f"Parameter '{name}' has no gradient"
            assert not torch.all(p.grad == 0), f"Parameter '{name}' has zero gradient"

    # ── train / eval modes ────────────────────────────────────

    def test_train_eval_modes_do_not_crash(self, device):
        """Forward pass works in both train() and eval() modes."""
        x = torch.randn(2, 256, 40, 40, device=device)
        blk = BottleneckDepth(256, 64).to(device)

        blk.train()
        out_train = blk(x)
        assert out_train.shape == (2, 256, 40, 40)

        blk.eval()
        with torch.no_grad():
            out_eval = blk(x)
        assert out_eval.shape == (2, 256, 40, 40)

    # ── All-zero input ────────────────────────────────────────

    def test_zero_input_does_not_crash(self, device):
        """All-zero input should not cause NaN/inf."""
        x = torch.zeros(2, 256, 40, 40, device=device)
        blk = BottleneckDepth(256, 64, stride=1).to(device)
        out = blk(x)
        assert torch.isfinite(out).all(), "Output should be finite for zero input"
