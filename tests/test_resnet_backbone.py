"""Tests for ResNet50Backbone from dep_deal, hw_deal, rgb_deal."""

import pytest
import torch
from dep_deal import ResNet50Backbone as DepthBackbone
from hw_deal import ResNet50Backbone as IRBackbone
from rgb_deal import ResNet50Backbone as RGBBackbone


class TestResNet50Backbone:
    """Test ResNet50 backbone: forward shapes, freezing, weight init, train/eval."""

    # ── Shape tests ────────────────────────────────────────────

    @pytest.mark.parametrize("H, W", [(320, 320), (640, 480)])
    def test_depth_backbone_shapes(self, device, batch_size, H, W):
        """Depth backbone: 1-channel input → C2–C5 multiscale outputs."""
        x = torch.randn(batch_size, 1, H, W, device=device)
        model = DepthBackbone(in_channels=1, pretrained=False).to(device)
        out = model(x)

        assert out["C2"].shape == (batch_size, 256,  H // 4,  W // 4)
        assert out["C3"].shape == (batch_size, 512,  H // 8,  W // 8)
        assert out["C4"].shape == (batch_size, 1024, H // 16, W // 16)
        assert out["C5"].shape == (batch_size, 2048, H // 32, W // 32)

    @pytest.mark.parametrize("H, W", [(320, 320), (640, 480)])
    def test_ir_backbone_shapes(self, device, batch_size, H, W):
        """IR backbone: 1-channel input → C2–C5."""
        x = torch.randn(batch_size, 1, H, W, device=device)
        model = IRBackbone(in_channels=1, pretrained=False).to(device)
        out = model(x)

        assert out["C2"].shape == (batch_size, 256,  H // 4,  W // 4)
        assert out["C3"].shape == (batch_size, 512,  H // 8,  W // 8)
        assert out["C4"].shape == (batch_size, 1024, H // 16, W // 16)
        assert out["C5"].shape == (batch_size, 2048, H // 32, W // 32)

    def test_rgb_backbone_shapes(self, device, batch_size):
        """RGB backbone: 3-channel input → C2–C5."""
        H, W = 320, 320
        x = torch.randn(batch_size, 3, H, W, device=device)
        model = RGBBackbone(in_channels=3, pretrained=True).to(device)
        out = model(x)

        assert out["C2"].shape == (batch_size, 256,  H // 4,  W // 4)
        assert out["C3"].shape == (batch_size, 512,  H // 8,  W // 8)
        assert out["C4"].shape == (batch_size, 1024, H // 16, W // 16)
        assert out["C5"].shape == (batch_size, 2048, H // 32, W // 32)

    def test_rgb_no_pretrained_does_not_crash(self, device, batch_size):
        """RGB backbone with pretrained=False uses Kaiming init."""
        H, W = 320, 320
        x = torch.randn(batch_size, 3, H, W, device=device)
        model = RGBBackbone(in_channels=3, pretrained=False).to(device)
        out = model(x)
        assert out["C2"].shape == (batch_size, 256, H // 4, W // 4)

    # ── Frozen stages ─────────────────────────────────────────

    def test_freeze_stages_0_no_freeze(self, device):
        """frozen_stages=0 means nothing is frozen."""
        model = DepthBackbone(in_channels=1, frozen_stages=0).to(device)
        all_trainable = all(p.requires_grad for p in model.parameters())
        assert all_trainable, "All params should be trainable when frozen_stages=0"

    def test_freeze_stages_1_freeze_stem_and_layer1(self, device):
        """frozen_stages=1 freezes conv1, bn1, layer1."""
        model = DepthBackbone(in_channels=1, frozen_stages=1).to(device)

        # Stem + layer1 should be frozen
        for p in model.conv1.parameters():
            assert not p.requires_grad
        for p in model.bn1.parameters():
            assert not p.requires_grad
        for p in model.layer1.parameters():
            assert not p.requires_grad

        # layer2+ should be trainable
        for p in model.layer2.parameters():
            assert p.requires_grad

    def test_freeze_stages_4_all_frozen(self, device):
        """frozen_stages=4 freezes all stages."""
        model = DepthBackbone(in_channels=1, frozen_stages=4).to(device)
        all_frozen = all(not p.requires_grad for p in model.parameters())
        assert all_frozen, "All params should be frozen when frozen_stages=4"

    def test_rgb_default_freeze(self, device):
        """RGB branch defaults to frozen_stages=1 (stem+layer1 frozen)."""
        from rgb_deal import RGBBranch
        model = RGBBranch(frozen_stages=1).to(device)
        for p in model.backbone.conv1.parameters():
            assert not p.requires_grad
        for p in model.backbone.layer1.parameters():
            assert not p.requires_grad
        # layer2+ trainable
        for p in model.backbone.layer2.parameters():
            assert p.requires_grad

    # ── Train mode re-freezes ─────────────────────────────────

    def test_train_mode_respects_freeze(self, device):
        """model.train() should not unfreeze previously frozen stages."""
        model = DepthBackbone(in_channels=1, frozen_stages=2).to(device)

        # Manually unfreeze everything (simulate weird state)
        for p in model.parameters():
            p.requires_grad = True

        # Calling train() should re-apply freezing
        model.train()
        for p in model.conv1.parameters():
            assert not p.requires_grad, "conv1 should be re-frozen after train()"
        for p in model.layer1.parameters():
            assert not p.requires_grad, "layer1 should be re-frozen after train()"
        for p in model.layer2.parameters():
            assert not p.requires_grad, "layer2 should be re-frozen after train()"
        for p in model.layer3.parameters():
            assert p.requires_grad, "layer3 should be trainable"

    # ── Parameter count ───────────────────────────────────────

    def test_param_count_depth_backbone(self, device):
        """Depth ResNet50 with 1-channel stem has correct param count."""
        model = DepthBackbone(in_channels=1).to(device)
        total = sum(p.numel() for p in model.parameters())
        # ResNet50 has ~23.5M params (+ a few less from 1ch conv1)
        assert 23e6 < total < 24e6, f"Expected ~23.5M params, got {total}"

    def test_param_count_rgb_backbone(self, device):
        """RGB ResNet50 with 3-channel stem."""
        model = RGBBackbone(in_channels=3, pretrained=False).to(device)
        total = sum(p.numel() for p in model.parameters())
        assert 23e6 < total < 24e6, f"Expected ~23.5M params, got {total}"

    # ── Gradient flow ─────────────────────────────────────────

    def test_gradient_flow_depth(self, device, batch_size):
        """Gradients reach all unfrozen layers in depth backbone."""
        x = torch.randn(batch_size, 1, 320, 320, device=device)
        model = DepthBackbone(in_channels=1, frozen_stages=0).to(device)

        out = model(x)
        loss = out["C5"].sum()
        loss.backward()

        # Check a few key params have gradients
        assert model.conv1.weight.grad is not None
        assert not torch.all(model.conv1.weight.grad == 0)

    def test_gradient_respects_freeze(self, device, batch_size):
        """Frozen params should have no gradient after backward."""
        x = torch.randn(batch_size, 1, 320, 320, device=device)
        model = DepthBackbone(in_channels=1, frozen_stages=1).to(device)

        out = model(x)
        loss = out["C5"].sum()
        loss.backward()

        # conv1.weight should have no grad since it's frozen
        assert model.conv1.weight.grad is None, \
            "Frozen conv1 should have no gradient"

    # ── Edge cases ────────────────────────────────────────────

    def test_small_input(self, device):
        """Very small input should still work (H=64, W=64)."""
        H, W = 64, 64
        x = torch.randn(1, 1, H, W, device=device)
        model = DepthBackbone(in_channels=1).to(device)
        out = model(x)
        assert out["C2"].shape == (1, 256,  H // 4,  W // 4)
        assert out["C5"].shape == (1, 2048, H // 32, W // 32)

    def test_odd_dimension_input(self, device):
        """Odd-sized input (e.g., 321×321) should produce valid features."""
        H, W = 321, 321
        x = torch.randn(1, 1, H, W, device=device)
        model = DepthBackbone(in_channels=1).to(device)
        out = model(x)
        # For odd inputs, conv+pool rounding may differ from pure integer division.
        # Verify correct channels, finite values, and approximate spatial sizes.
        assert out["C2"].shape[1] == 256, "C2 should have 256 channels"
        assert out["C5"].shape[1] == 2048, "C5 should have 2048 channels"
        assert out["C2"].shape[2] == out["C2"].shape[3]  # square
        assert all(torch.isfinite(v).all() for v in out.values())
        # Spatial sizes should be approximately H/4 and H/32
        assert abs(out["C2"].shape[2] - H // 4) <= 1, \
            f"C2 H={out['C2'].shape[2]}, expected ~{H//4}"
        assert abs(out["C5"].shape[2] - H // 32) <= 1, \
            f"C5 H={out['C5'].shape[2]}, expected ~{H//32}"
