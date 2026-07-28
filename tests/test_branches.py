"""Tests for DepthBranch, InfraredBranch, RGBBranch — sensor-specific extractors."""

import pytest
import torch
from dep_deal import DepthBranch
from hw_deal import InfraredBranch
from rgb_deal import RGBBranch


class TestDepthBranch:
    """Depth branch: single-channel input, ResNet50 trained from scratch."""

    def test_output_shapes(self, depth_input, device):
        """Output is a dict with C2–C5 at correct spatial scales."""
        model = DepthBranch(frozen_stages=-1).to(device)
        out = model(depth_input)

        B, _, H, W = depth_input.shape
        assert out["C2"].shape == (B, 256,  H // 4,  W // 4)
        assert out["C3"].shape == (B, 512,  H // 8,  W // 8)
        assert out["C4"].shape == (B, 1024, H // 16, W // 16)
        assert out["C5"].shape == (B, 2048, H // 32, W // 32)

    def test_default_no_freeze(self, device, depth_input):
        """By default (frozen_stages=-1), all params are trainable."""
        model = DepthBranch(frozen_stages=-1).to(device)
        out = model(depth_input)
        loss = out["C5"].sum()
        loss.backward()
        assert model.backbone.conv1.weight.grad is not None, \
            "conv1 should have gradient when no freeze"

    def test_freeze_stages_1(self, device, depth_input):
        """frozen_stages=1 freezes stem and layer1."""
        model = DepthBranch(frozen_stages=1).to(device)
        assert not model.backbone.conv1.weight.requires_grad
        assert model.backbone.layer2[0].conv1.weight.requires_grad

    def test_param_count(self, device):
        """DepthBranch has ~23.5M params."""
        model = DepthBranch().to(device)
        total = sum(p.numel() for p in model.parameters())
        assert 23e6 < total < 24e6

    def test_trainable_count(self, device):
        """All params trainable by default."""
        model = DepthBranch(frozen_stages=-1).to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable == total


class TestInfraredBranch:
    """Infrared branch: structurally identical to DepthBranch."""

    def test_output_shapes(self, thermal_input, device):
        """IR branch outputs same structure as depth branch."""
        model = InfraredBranch(frozen_stages=-1).to(device)
        out = model(thermal_input)

        B, _, H, W = thermal_input.shape
        assert out["C2"].shape == (B, 256,  H // 4,  W // 4)
        assert out["C3"].shape == (B, 512,  H // 8,  W // 8)
        assert out["C4"].shape == (B, 1024, H // 16, W // 16)
        assert out["C5"].shape == (B, 2048, H // 32, W // 32)

    def test_default_no_freeze(self, device, thermal_input):
        """Default: no freezing, gradients flow everywhere."""
        model = InfraredBranch(frozen_stages=-1).to(device)
        out = model(thermal_input)
        loss = out["C5"].sum()
        loss.backward()
        assert model.backbone.conv1.weight.grad is not None

    def test_depth_and_ir_independent(self, device, depth_input, thermal_input):
        """Depth and IR branches produce different features from same input."""
        model_d = DepthBranch().to(device)
        model_ir = InfraredBranch().to(device)

        # Same random tensor as input to both
        shared_input = depth_input  # (B, 1, H, W)

        with torch.no_grad():
            out_d = model_d(shared_input)
            out_ir = model_ir(shared_input)

        # Different random init → different outputs
        assert not torch.allclose(out_d["C5"], out_ir["C5"]), \
            "Depth and IR branches should differ with random init"


class TestRGBBranch:
    """RGB branch: 3-channel input, ImageNet pretrained, frozen stem+layer1."""

    def test_output_shapes(self, rgb_input, device):
        """RGB branch outputs C2–C5."""
        model = RGBBranch(frozen_stages=1).to(device)
        out = model(rgb_input)

        B, _, H, W = rgb_input.shape
        assert out["C2"].shape == (B, 256,  H // 4,  W // 4)
        assert out["C3"].shape == (B, 512,  H // 8,  W // 8)
        assert out["C4"].shape == (B, 1024, H // 16, W // 16)
        assert out["C5"].shape == (B, 2048, H // 32, W // 32)

    def test_pretrained_weights_loaded(self, device):
        """ImageNet pretrained weights should be loaded when pretrained=True."""
        model = RGBBranch(frozen_stages=1).to(device)
        # Check BN weight in deeper layers: should not be default 1.0
        bn_weight = model.backbone.layer4[-1].bn3.weight
        # Pretrained BN weight should differ from Kaiming init default of 1.0
        assert not torch.allclose(bn_weight, torch.ones_like(bn_weight)), \
            "layer4.bn3.weight should contain pretrained weights, not default ones"

    def test_default_freeze_stem(self, device):
        """Default frozen_stages=1 freezes conv1, bn1, layer1."""
        model = RGBBranch().to(device)
        assert not model.backbone.conv1.weight.requires_grad, \
            "conv1 should be frozen by default"
        assert not model.backbone.layer1[0].conv1.weight.requires_grad, \
            "layer1 should be frozen by default"
        assert model.backbone.layer2[0].conv1.weight.requires_grad, \
            "layer2 should be trainable by default"

    def test_no_freeze_option(self, device):
        """frozen_stages=-1 allows training all layers."""
        model = RGBBranch(frozen_stages=-1).to(device)
        all_trainable = all(p.requires_grad for p in model.parameters())
        assert all_trainable

    def test_trainable_vs_total(self, device):
        """With frozen_stages=1, only ~60% params are trainable."""
        model = RGBBranch(frozen_stages=1).to(device)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable < total, "Some params should be frozen"
        assert trainable > 0.5 * total, "Most params should still be trainable"

    @pytest.mark.parametrize("frozen", [-1, 0, 1, 2, 3, 4])
    def test_frozen_stages_forward(self, frozen, rgb_input, device):
        """Forward pass succeeds for all frozen_stages values."""
        model = RGBBranch(frozen_stages=frozen).to(device)
        out = model(rgb_input)
        B, _, H, W = rgb_input.shape
        assert out["C2"].shape == (B, 256, H // 4, W // 4)
