"""Tests for FPN, RGBFPN, DTFPN — Feature Pyramid Networks."""

import pytest
import torch
from rgb_fpn import FPN, RGBFPN
from dt_fpn import DTFPN


# ───────────────────────────────────────────────────────────────
# FPN 核心
# ───────────────────────────────────────────────────────────────
class TestFPN:
    """Test the core FPN module."""

    IN_CH = {"C2": 256, "C3": 512, "C4": 1024, "C5": 2048}
    OUT_CH = 256

    @pytest.fixture
    def fpn(self, device):
        return FPN(in_channels=self.IN_CH, out_channels=self.OUT_CH).to(device)

    def test_output_shapes(self, fpn, multiscale_feats):
        """Output P2–P5 have out_channels and correct spatial dims."""
        out = fpn(multiscale_feats)
        B = multiscale_feats["C2"].shape[0]

        assert out["P2"].shape == (B, self.OUT_CH, *multiscale_feats["C2"].shape[2:])
        assert out["P3"].shape == (B, self.OUT_CH, *multiscale_feats["C3"].shape[2:])
        assert out["P4"].shape == (B, self.OUT_CH, *multiscale_feats["C4"].shape[2:])
        assert out["P5"].shape == (B, self.OUT_CH, *multiscale_feats["C5"].shape[2:])

    def test_all_levels_present(self, fpn, multiscale_feats):
        """Output should have exactly P2, P3, P4, P5."""
        out = fpn(multiscale_feats)
        assert set(out.keys()) == {"P2", "P3", "P4", "P5"}

    def test_output_finite(self, fpn, multiscale_feats):
        """All output features should be finite."""
        out = fpn(multiscale_feats)
        for level, feat in out.items():
            assert torch.isfinite(feat).all(), f"{level} has NaN/Inf"

    def test_gradient_flow(self, fpn, multiscale_feats, device):
        """Gradients should flow to all trainable params."""
        fpn.train()
        out = fpn(multiscale_feats)
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        for name, p in fpn.named_parameters():
            assert p.grad is not None, f"'{name}' has no grad"
            assert not torch.all(p.grad == 0), f"'{name}' has zero grad"

    def test_top_down_feature_propagation(self, device):
        """Features should propagate top-down: P4 depends on P5 (via upsample)."""
        # Create a custom FPN with fixed weights to verify propagation
        in_ch = {"C4": 256, "C5": 256}
        fpn = FPN(in_channels=in_ch, out_channels=64).to(device)
        fpn.eval()

        # Zero all weights so lateral convs output zero
        with torch.no_grad():
            for p in fpn.parameters():
                p.zero_()

        # C4 non-zero, C5 zero → with zero weights:
        # lateral_C5 = 0, P5 = conv(lateral_C5) = 0
        # lateral_C4 = 0, upsampled_P5 = 0, P4 = conv(0+0) = 0
        B = 2
        feats = {
            "C4": torch.randn(B, 256, 40, 40, device=device),
            "C5": torch.randn(B, 256, 20, 20, device=device),
        }
        out = fpn(feats)
        # With all-zero weights, everything should be zero
        assert torch.allclose(out["P4"], torch.zeros_like(out["P4"]), atol=1e-6)
        assert torch.allclose(out["P5"], torch.zeros_like(out["P5"]), atol=1e-6)

    def test_nearest_upsample_integration(self, device):
        """Nearest-neighbor upsampling should preserve spatial integrity."""
        in_ch = {"C4": 128, "C5": 128}
        fpn = FPN(in_channels=in_ch, out_channels=128).to(device)
        fpn.eval()

        B, H4, W4, H5, W5 = 2, 40, 40, 20, 20
        feats = {
            "C4": torch.randn(B, 128, H4, W4, device=device),
            "C5": torch.randn(B, 128, H5, W5, device=device),
        }
        out = fpn(feats)
        # P4 spatial should match C4 (H4, W4)
        assert out["P4"].shape[2:] == (H4, W4)
        assert out["P5"].shape[2:] == (H5, W5)

    def test_param_count(self, fpn):
        """FPN has modest param count."""
        total = sum(p.numel() for p in fpn.parameters())
        assert 2e6 < total < 10e6, f"FPN params {total} outside expected range"

    def test_train_eval_modes(self, fpn, multiscale_feats):
        """Forward pass works in both train/eval."""
        fpn.train()
        out_train = fpn(multiscale_feats)

        fpn.eval()
        with torch.no_grad():
            out_eval = fpn(multiscale_feats)

        for k in out_train:
            assert out_train[k].shape == out_eval[k].shape

    def test_extra_level_p6(self, multiscale_feats, device):
        """extra_level=True adds P6 at stride=64."""
        fpn = FPN(in_channels=self.IN_CH, out_channels=256,
                  extra_level=True).to(device)
        out = fpn(multiscale_feats)
        assert "P6" in out
        # P6 should be P5 downsampled by 2
        B, _, H5, W5 = multiscale_feats["C5"].shape
        assert out["P6"].shape == (B, 256, H5 // 2, W5 // 2)

    def test_different_out_channels(self, multiscale_feats, device):
        """Can specify different output channel count."""
        fpn = FPN(in_channels=self.IN_CH, out_channels=128).to(device)
        out = fpn(multiscale_feats)
        for v in out.values():
            assert v.shape[1] == 128

    def test_partial_scales(self, device):
        """FPN works with only C4+C5 (not all C2–C5)."""
        in_ch = {"C4": 1024, "C5": 2048}
        B = 2
        feats = {
            "C4": torch.randn(B, 1024, 20, 20, device=device),
            "C5": torch.randn(B, 2048, 10, 10, device=device),
        }
        fpn = FPN(in_channels=in_ch).to(device)
        out = fpn(feats)
        assert set(out.keys()) == {"P4", "P5"}


# ───────────────────────────────────────────────────────────────
# RGBFPN
# ───────────────────────────────────────────────────────────────
class TestRGBFPN:
    """Test RGB-specific FPN wrapper."""

    def test_output_shapes(self, multiscale_feats, device):
        """RGBFPN wraps FPN — same output semantics."""
        model = RGBFPN(out_channels=256).to(device)
        out = model(multiscale_feats)

        B = multiscale_feats["C2"].shape[0]
        assert out["P2"].shape == (B, 256, *multiscale_feats["C2"].shape[2:])
        assert out["P3"].shape == (B, 256, *multiscale_feats["C3"].shape[2:])
        assert out["P4"].shape == (B, 256, *multiscale_feats["C4"].shape[2:])
        assert out["P5"].shape == (B, 256, *multiscale_feats["C5"].shape[2:])

    def test_param_count(self, device):
        """RGBFPN parameter count."""
        model = RGBFPN().to(device)
        total = sum(p.numel() for p in model.parameters())
        assert 2e6 < total < 10e6

    def test_forward_twice_same_result_eval(self, multiscale_feats, device):
        """Deterministic in eval mode."""
        model = RGBFPN().to(device).eval()
        with torch.no_grad():
            out1 = model(multiscale_feats)
            out2 = model(multiscale_feats)
        for k in out1:
            assert torch.equal(out1[k], out2[k]), f"{k} differs between two eval passes"


# ───────────────────────────────────────────────────────────────
# DTFPN
# ───────────────────────────────────────────────────────────────
class TestDTFPN:
    """Test DT-specific FPN wrapper."""

    def test_output_shapes(self, multiscale_feats, device):
        """DTFPN output same structure as RGBFPN."""
        model = DTFPN(out_channels=256).to(device)
        out = model(multiscale_feats)

        B = multiscale_feats["C2"].shape[0]
        assert out["P2"].shape == (B, 256, *multiscale_feats["C2"].shape[2:])
        assert out["P3"].shape == (B, 256, *multiscale_feats["C3"].shape[2:])
        assert out["P4"].shape == (B, 256, *multiscale_feats["C4"].shape[2:])
        assert out["P5"].shape == (B, 256, *multiscale_feats["C5"].shape[2:])

    def test_param_count(self, device):
        """DTFPN has same param count as RGBFPN."""
        model = DTFPN().to(device)
        total = sum(p.numel() for p in model.parameters())
        assert 2e6 < total < 10e6

    def test_rgb_fpn_dt_fpn_independent(self, multiscale_feats, device):
        """RGBFPN and DTFPN are separate instances with independent params."""
        rgb_fpn = RGBFPN().to(device)
        dt_fpn = DTFPN().to(device)

        with torch.no_grad():
            out_rgb = rgb_fpn(multiscale_feats)
            out_dt = dt_fpn(multiscale_feats)

        # Different random init → different outputs
        for k in out_rgb:
            assert not torch.allclose(out_rgb[k], out_dt[k]), \
                f"RGBFPN and DTFPN should differ at {k} with random init"

    def test_gradient_flow(self, multiscale_feats, device):
        """Gradients flow through DTFPN."""
        model = DTFPN().to(device)
        model.train()
        out = model(multiscale_feats)
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"'{name}' has no grad"
