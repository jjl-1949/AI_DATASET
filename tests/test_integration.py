"""End-to-end integration tests — full forward pipeline."""

import pytest
import torch

from dep_deal import DepthBranch
from hw_deal import InfraredBranch
from rgb_deal import RGBBranch
from dt_ort import DTOrtFusion
from rgb_fpn import RGBFPN
from dt_fpn import DTFPN


class TestDepthThermalPipeline:
    """Depth + Thermal → DTOrtFusion → DTFPN."""

    @pytest.fixture
    def dt_pipeline(self, device):
        """Build the Depth–Thermal pipeline."""
        depth_branch = DepthBranch(frozen_stages=-1).to(device)
        ir_branch = InfraredBranch(frozen_stages=-1).to(device)
        dt_fusion = DTOrtFusion().to(device)
        dt_fpn = DTFPN(out_channels=256).to(device)
        return depth_branch, ir_branch, dt_fusion, dt_fpn

    def test_full_forward(self, dt_pipeline, depth_input, thermal_input):
        """Complete depth+thermal → fusion → FPN forward pass."""
        depth_branch, ir_branch, dt_fusion, dt_fpn = dt_pipeline

        # Extract features
        d_feats = depth_branch(depth_input)
        t_feats = ir_branch(thermal_input)

        # Orthogonal fusion
        fused = dt_fusion(d_feats, t_feats)

        # FPN
        pyramid = dt_fpn(fused)

        B, _, H, W = depth_input.shape
        assert pyramid["P2"].shape == (B, 256, H // 4,  W // 4)
        assert pyramid["P3"].shape == (B, 256, H // 8,  W // 8)
        assert pyramid["P4"].shape == (B, 256, H // 16, W // 16)
        assert pyramid["P5"].shape == (B, 256, H // 32, W // 32)

    def test_gradient_flow_full_pipeline(self, dt_pipeline, depth_input, thermal_input):
        """Gradients flow end-to-end through the DT pipeline."""
        depth_branch, ir_branch, dt_fusion, dt_fpn = dt_pipeline

        # Enable training
        for m in [depth_branch, ir_branch, dt_fusion, dt_fpn]:
            m.train()

        d_feats = depth_branch(depth_input)
        t_feats = ir_branch(thermal_input)
        fused = dt_fusion(d_feats, t_feats)
        pyramid = dt_fpn(fused)

        loss = pyramid["P5"].sum()
        loss.backward()

        # Check gradient reaches the first layer
        assert depth_branch.backbone.conv1.weight.grad is not None, \
            "Gradient should reach depth conv1"
        assert ir_branch.backbone.conv1.weight.grad is not None, \
            "Gradient should reach IR conv1"

    def test_output_finite(self, dt_pipeline, depth_input, thermal_input):
        """All outputs should be finite."""
        depth_branch, ir_branch, dt_fusion, dt_fpn = dt_pipeline

        with torch.no_grad():
            d_feats = depth_branch(depth_input)
            t_feats = ir_branch(thermal_input)
            fused = dt_fusion(d_feats, t_feats)
            pyramid = dt_fpn(fused)

        for name, feat in {**fused, **pyramid}.items():
            assert torch.isfinite(feat).all(), f"{name} has NaN/Inf"


class TestRGBPipeline:
    """RGB → RGBFPN."""

    @pytest.fixture
    def rgb_pipeline(self, device):
        """Build RGB pipeline."""
        rgb_branch = RGBBranch(frozen_stages=1).to(device)
        rgb_fpn = RGBFPN(out_channels=256).to(device)
        return rgb_branch, rgb_fpn

    def test_full_forward(self, rgb_pipeline, rgb_input):
        """RGB → FPN forward pass."""
        rgb_branch, rgb_fpn = rgb_pipeline

        feats = rgb_branch(rgb_input)
        pyramid = rgb_fpn(feats)

        B, _, H, W = rgb_input.shape
        assert pyramid["P2"].shape == (B, 256, H // 4,  W // 4)
        assert pyramid["P3"].shape == (B, 256, H // 8,  W // 8)
        assert pyramid["P4"].shape == (B, 256, H // 16, W // 16)
        assert pyramid["P5"].shape == (B, 256, H // 32, W // 32)

    def test_gradient_flow(self, rgb_pipeline, rgb_input):
        """Gradients flow through unfrozen layers."""
        rgb_branch, rgb_fpn = rgb_pipeline

        rgb_branch.train()
        rgb_fpn.train()

        feats = rgb_branch(rgb_input)
        pyramid = rgb_fpn(feats)

        loss = pyramid["P5"].sum()
        loss.backward()

        # Stem + layer1 frozen → no gradient
        assert rgb_branch.backbone.conv1.weight.grad is None, \
            "conv1 should have no grad (frozen)"

        # layer4 should have gradient
        assert rgb_branch.backbone.layer4[0].conv1.weight.grad is not None, \
            "layer4 should have gradient"


class TestThreeSensorPipeline:
    """Full three-sensor pipeline: RGB + Depth + Thermal."""

    def test_full_forward(self, device, rgb_input, depth_input, thermal_input):
        """All 3 branches → 2 FPNs → outputs align per scale."""
        # Build all modules
        rgb_branch = RGBBranch(frozen_stages=1).to(device)
        depth_branch = DepthBranch().to(device)
        ir_branch = InfraredBranch().to(device)
        dt_fusion = DTOrtFusion().to(device)
        rgb_fpn = RGBFPN().to(device)
        dt_fpn = DTFPN().to(device)

        with torch.no_grad():
            rgb_feats = rgb_branch(rgb_input)
            d_feats = depth_branch(depth_input)
            t_feats = ir_branch(thermal_input)

            dt_fused = dt_fusion(d_feats, t_feats)

            rgb_pyramid = rgb_fpn(rgb_feats)
            dt_pyramid = dt_fpn(dt_fused)

        # Both pyramids should have matching spatial dimensions per scale
        for level in ["P2", "P3", "P4", "P5"]:
            assert rgb_pyramid[level].shape == dt_pyramid[level].shape, \
                f"{level}: RGB {tuple(rgb_pyramid[level].shape)} != DT {tuple(dt_pyramid[level].shape)}"

    def test_param_count_summary(self, device):
        """Print total parameter counts for the full pipeline."""
        modules = {
            "RGBBranch": RGBBranch(frozen_stages=1),
            "DepthBranch": DepthBranch(),
            "InfraredBranch": InfraredBranch(),
            "DTOrtFusion": DTOrtFusion(),
            "RGBFPN": RGBFPN(),
            "DTFPN": DTFPN(),
        }

        for name, m in modules.items():
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"  {name}: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")

        grand_total = sum(sum(p.numel() for p in m.parameters()) for m in modules.values())
        print(f"  TOTAL: {grand_total/1e6:.2f}M")
        # Just ensure it's in a reasonable range
        assert 50e6 < grand_total < 200e6, \
            f"Total params {grand_total} outside expected range [50M, 200M]"

    def test_different_input_sizes(self, device):
        """Pipeline handles different input sizes."""
        sizes = [(320, 320), (640, 480), (256, 256)]
        for H, W in sizes:
            rgb = torch.randn(1, 3, H, W, device=device)
            depth = torch.randn(1, 1, H, W, device=device)
            ir = torch.randn(1, 1, H, W, device=device)

            rgb_branch = RGBBranch(frozen_stages=1).to(device)
            depth_branch = DepthBranch().to(device)
            ir_branch = InfraredBranch().to(device)
            dt_fusion = DTOrtFusion().to(device)
            rgb_fpn = RGBFPN().to(device)
            dt_fpn = DTFPN().to(device)

            with torch.no_grad():
                rgb_feats = rgb_branch(rgb)
                d_feats = depth_branch(depth)
                t_feats = ir_branch(ir)
                fused = dt_fusion(d_feats, t_feats)
                rgb_pyr = rgb_fpn(rgb_feats)
                dt_pyr = dt_fpn(fused)

            # All outputs should be finite
            for name, feat in {**rgb_pyr, **dt_pyr}.items():
                assert torch.isfinite(feat).all(), \
                    f"Input {H}×{W}, {name}: NaN/Inf detected"


class TestMemoryAndDeterminism:
    """Cross-cutting concerns: determinism & memory."""

    def test_deterministic_in_eval(self, device, rgb_input, depth_input, thermal_input):
        """Full pipeline in eval is deterministic."""
        torch.manual_seed(0)

        def run_pipeline():
            rgb_branch = RGBBranch(frozen_stages=1).to(device).eval()
            depth_branch = DepthBranch().to(device).eval()
            ir_branch = InfraredBranch().to(device).eval()
            dt_fusion = DTOrtFusion().to(device).eval()
            rgb_fpn = RGBFPN().to(device).eval()
            dt_fpn = DTFPN().to(device).eval()

            with torch.no_grad():
                rgb_feats = rgb_branch(rgb_input)
                d_feats = depth_branch(depth_input)
                t_feats = ir_branch(thermal_input)
                fused = dt_fusion(d_feats, t_feats)
                rgb_pyr = rgb_fpn(rgb_feats)
                dt_pyr = dt_fpn(fused)
            return dt_pyr["P5"]

        torch.manual_seed(0)
        out1 = run_pipeline()
        torch.manual_seed(0)
        out2 = run_pipeline()
        assert torch.equal(out1, out2), "Pipeline should be deterministic with fixed seed"
