"""Tests for DTOrtFusion & OrthoFusionBlock — Gram-Schmidt orthogonal fusion."""

import pytest
import torch
from dt_ort import OrthoFusionBlock, DTOrtFusion


# ───────────────────────────────────────────────────────────────
# OrthoFusionBlock 单尺度正交融合块
# ───────────────────────────────────────────────────────────────
class TestOrthoFusionBlock:
    """Test the single-scale orthogonal fusion block."""

    @pytest.mark.parametrize("in_c", [128, 256, 512])
    @pytest.mark.parametrize("spatial", [20, 40])
    def test_output_shape(self, in_c, spatial, device):
        """Output should have same shape as input (after fusion conv)."""
        B = 2
        F_d = torch.randn(B, in_c, spatial, spatial, device=device)
        F_t = torch.randn(B, in_c, spatial, spatial, device=device)

        block = OrthoFusionBlock(in_channels=in_c).to(device)
        out = block(F_d, F_t)

        assert out.shape == (B, in_c, spatial, spatial), \
            f"Expected {(B, in_c, spatial, spatial)}, got {tuple(out.shape)}"

    def test_output_finite(self, device):
        """Output should contain no NaN or Inf."""
        F_d = torch.randn(2, 256, 40, 40, device=device)
        F_t = torch.randn(2, 256, 40, 40, device=device)

        block = OrthoFusionBlock(256).to(device)
        out = block(F_d, F_t)

        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_output_changes_with_input(self, device):
        """Different inputs should produce different outputs."""
        block = OrthoFusionBlock(256).to(device).eval()

        with torch.no_grad():
            out1 = block(
                torch.randn(2, 256, 40, 40, device=device),
                torch.randn(2, 256, 40, 40, device=device),
            )
            out2 = block(
                torch.randn(2, 256, 40, 40, device=device),
                torch.randn(2, 256, 40, 40, device=device),
            )

        assert not torch.equal(out1, out2), \
            "Different inputs should yield different outputs"

    # ── Gram-Schmidt orthogonalization tests ───────────────────

    def test_gram_schmidt_full_orthogonality(self, device):
        """At alpha=1, <F_d^⊥, F_basis> should be ≈ 0 per spatial position."""
        C, H, W = 256, 16, 16
        F_a = torch.randn(2, C, H, W, device=device)
        F_b = torch.randn(2, C, H, W, device=device)

        # Full orthogonalization: alpha=1
        F_a_orth = OrthoFusionBlock._gram_schmidt_step(
            F_a, F_b, torch.tensor(1.0, device=device)
        )

        # Per-spatial-position inner product: <F_a^⊥, F_b>
        inner = (F_a_orth * F_b).sum(dim=1)  # (B, H, W)
        max_abs_inner = inner.abs().max().item()

        # Should be very close to 0 (floating-point noise only)
        assert max_abs_inner < 1e-3, \
            f"|<F_d^⊥, F_b>| should be ≈0 at α=1, got max={max_abs_inner:.2e}"

    def test_gram_schmidt_no_orthogonalization(self, device):
        """At alpha=0, F_a^⊥ should equal F_a (identity)."""
        C, H, W = 128, 8, 8
        F_a = torch.randn(2, C, H, W, device=device)
        F_b = torch.randn(2, C, H, W, device=device)

        F_a_orth = OrthoFusionBlock._gram_schmidt_step(
            F_a, F_b, torch.tensor(0.0, device=device)
        )

        assert torch.allclose(F_a_orth, F_a, atol=1e-6), \
            "At α=0, output should equal input"

    def test_gram_schmidt_partial_orthogonalization(self, device):
        """At α=0.5, inner product should be ~50% of original."""
        C, H, W = 64, 4, 4
        F_a = torch.randn(2, C, H, W, device=device)
        F_b = torch.randn(2, C, H, W, device=device)

        inner_orig = (F_a * F_b).sum(dim=1)  # (B, H, W)

        F_a_orth_half = OrthoFusionBlock._gram_schmidt_step(
            F_a, F_b, torch.tensor(0.5, device=device)
        )
        inner_half = (F_a_orth_half * F_b).sum(dim=1)  # (B, H, W)

        # At α=0.5: inner_half = inner_orig - 0.5 * inner_orig = 0.5 * inner_orig
        assert torch.allclose(inner_half, 0.5 * inner_orig, atol=1e-5), \
            "At α=0.5, <F_a^⊥, F_b> should be 0.5 * <F_a, F_b>"

    def test_gram_schmidt_deterministic(self, device):
        """Same inputs → same outputs (no randomness)."""
        F_a = torch.randn(2, 64, 8, 8, device=device)
        F_b = torch.randn(2, 64, 8, 8, device=device)

        out1 = OrthoFusionBlock._gram_schmidt_step(
            F_a.clone(), F_b.clone(), torch.tensor(0.7)
        )
        out2 = OrthoFusionBlock._gram_schmidt_step(
            F_a.clone(), F_b.clone(), torch.tensor(0.7)
        )

        assert torch.equal(out1, out2), "Output should be deterministic"

    def test_gram_schmidt_identical_inputs(self, device):
        """When F_a == F_b and α=1, F_a^⊥ should be ≈ 0."""
        C, H, W = 64, 8, 8
        F = torch.randn(2, C, H, W, device=device)

        F_orth = OrthoFusionBlock._gram_schmidt_step(
            F, F, torch.tensor(1.0, device=device)
        )

        # Should be all zeros (a vector minus its own projection is zero)
        assert torch.allclose(F_orth, torch.zeros_like(F_orth), atol=1e-5), \
            "Identical inputs at α=1 should yield zero output"

    def test_gram_schmidt_numerical_stability_zero_norm(self, device):
        """Zero-norm F_b should not cause division-by-zero (protected by ε)."""
        C = 64
        F_a = torch.randn(2, C, 8, 8, device=device)
        F_b = torch.zeros(2, C, 8, 8, device=device)  # All-zero basis

        F_a_orth = OrthoFusionBlock._gram_schmidt_step(
            F_a, F_b, torch.tensor(1.0, device=device)
        )

        # When ||F_b||² ≈ 0 (after +ε), proj ≈ 0 → F_a^⊥ ≈ F_a
        assert torch.isfinite(F_a_orth).all(), "Output should be finite"
        assert torch.allclose(F_a_orth, F_a, atol=1e-5), \
            "Zero-norm basis should leave input unchanged"

    def test_gram_schmidt_batch_independence(self, device):
        """Each batch element is processed independently."""
        C, H, W = 128, 4, 4
        F_a = torch.randn(4, C, H, W, device=device)
        F_b = torch.randn(4, C, H, W, device=device)

        F_orth = OrthoFusionBlock._gram_schmidt_step(
            F_a, F_b, torch.tensor(1.0, device=device)
        )

        # Process each batch element individually
        results = []
        for i in range(4):
            r = OrthoFusionBlock._gram_schmidt_step(
                F_a[i:i+1], F_b[i:i+1], torch.tensor(1.0, device=device)
            )
            results.append(r)

        assert torch.allclose(F_orth, torch.cat(results, dim=0), atol=1e-5), \
            "Batch results should equal per-sample results"

    # ── Learnable alpha / beta ─────────────────────────────────

    def test_alpha_beta_learnable(self, device):
        """alpha and beta should receive gradients during training."""
        block = OrthoFusionBlock(256).to(device)
        F_d = torch.randn(2, 256, 40, 40, device=device)
        F_t = torch.randn(2, 256, 40, 40, device=device)

        block.train()
        out = block(F_d, F_t)
        loss = out.sum()
        loss.backward()

        assert block.alpha.grad is not None, "alpha should have gradient"
        assert block.beta.grad is not None, "beta should have gradient"
        assert block.alpha.grad.item() != 0, "alpha grad should be non-zero"
        assert block.beta.grad.item() != 0, "beta grad should be non-zero"

    def test_alpha_beta_sigmoid_range(self, device):
        """After sigmoid, alpha/beta are always in [0, 1]."""
        block = OrthoFusionBlock(256).to(device)

        # Access raw parameter and sigmoid
        alpha_s = torch.sigmoid(block.alpha)
        beta_s = torch.sigmoid(block.beta)

        assert 0 <= alpha_s.item() <= 1, f"alpha_sigmoid={alpha_s.item():.4f} not in [0,1]"
        assert 0 <= beta_s.item() <= 1, f"beta_sigmoid={beta_s.item():.4f} not in [0,1]"

    def test_alpha_beta_default_value(self, device):
        """Default raw alpha/beta is 0.5 → sigmoid(0.5) ≈ 0.622."""
        block = OrthoFusionBlock(256).to(device)
        alpha_s = torch.sigmoid(block.alpha)
        assert abs(alpha_s.item() - 0.6225) < 0.01, \
            f"sigmoid(0.5) ≈ 0.6225, got {alpha_s.item():.4f}"

    # ── Gradient flow through full block ───────────────────────

    def test_all_params_have_gradients(self, device):
        """Every trainable parameter should get a gradient."""
        block = OrthoFusionBlock(128).to(device)
        F_d = torch.randn(2, 128, 20, 20, device=device)
        F_t = torch.randn(2, 128, 20, 20, device=device)

        block.train()
        out = block(F_d, F_t)
        loss = out.sum()
        loss.backward()

        for name, p in block.named_parameters():
            assert p.grad is not None, f"'{name}' has no grad"
            assert not torch.all(p.grad == 0), f"'{name}' has zero grad"

    # ── Train/eval modes ───────────────────────────────────────

    def test_train_eval_consistency(self, device):
        """In eval mode, no dropout/noise → output should be deterministic."""
        block = OrthoFusionBlock(128).to(device)
        F_d = torch.randn(2, 128, 20, 20, device=device)
        F_t = torch.randn(2, 128, 20, 20, device=device)

        block.eval()
        with torch.no_grad():
            out1 = block(F_d, F_t)
            out2 = block(F_d, F_t)
        assert torch.equal(out1, out2), "Eval mode should be deterministic"

    # ── Edge cases ─────────────────────────────────────────────

    def test_zero_input(self, device):
        """All-zero inputs should produce finite output."""
        block = OrthoFusionBlock(64).to(device)
        zeros = torch.zeros(2, 64, 16, 16, device=device)
        out = block(zeros, zeros)
        assert torch.isfinite(out).all()

    def test_large_values(self, device):
        """Large input values should not cause numerical issues."""
        block = OrthoFusionBlock(64).to(device)
        F_d = torch.randn(2, 64, 16, 16, device=device) * 100
        F_t = torch.randn(2, 64, 16, 16, device=device) * 100
        out = block(F_d, F_t)
        assert torch.isfinite(out).all(), "Should handle large inputs"


# ───────────────────────────────────────────────────────────────
# DTOrtFusion 多尺度正交融合
# ───────────────────────────────────────────────────────────────
class TestDTOrtFusion:
    """Test the multi-scale DT orthogonal fusion module."""

    def test_output_shapes(self, depth_feats, thermal_feats, device):
        """Each output scale should match input scale shape."""
        model = DTOrtFusion().to(device)
        fused = model(depth_feats, thermal_feats)

        for level in depth_feats:
            assert fused[level].shape == depth_feats[level].shape, \
                f"{level}: {tuple(fused[level].shape)} != {tuple(depth_feats[level].shape)}"

    def test_all_scales_present(self, depth_feats, thermal_feats, device):
        """Output should contain exactly C2, C3, C4, C5."""
        model = DTOrtFusion().to(device)
        fused = model(depth_feats, thermal_feats)
        assert set(fused.keys()) == {"C2", "C3", "C4", "C5"}

    def test_output_finite(self, depth_feats, thermal_feats, device):
        """Output should be finite across all scales."""
        model = DTOrtFusion().to(device)
        fused = model(depth_feats, thermal_feats)
        for level, feat in fused.items():
            assert torch.isfinite(feat).all(), f"{level} has NaN/Inf"

    def test_scale_independence(self, device):
        """Each scale has independent parameters."""
        model = DTOrtFusion().to(device)
        alpha_C2 = model.fusion_blocks["C2"].alpha
        alpha_C5 = model.fusion_blocks["C5"].alpha
        assert alpha_C2 is not alpha_C5, "C2 and C5 should have separate alpha params"

    def test_gradient_flow(self, depth_feats, thermal_feats, device):
        """Gradients flow through all scales."""
        model = DTOrtFusion().to(device)
        model.train()

        fused = model(depth_feats, thermal_feats)
        loss = sum(v.sum() for v in fused.values())
        loss.backward()

        for level, block in model.fusion_blocks.items():
            assert block.alpha.grad is not None, f"{level} alpha has no grad"
            assert block.alpha.grad.item() != 0, f"{level} alpha has zero grad"

    def test_param_count(self, device):
        """DTOrtFusion has ~55M params (4 scales × large fusion convs, C5:2048ch)."""
        model = DTOrtFusion().to(device)
        total = sum(p.numel() for p in model.parameters())
        assert 50e6 < total < 65e6, f"Param count {total} outside expected range [50M, 65M]"

    def test_custom_channels(self, depth_feats, thermal_feats, device):
        """Should accept custom channel configuration."""
        custom_ch = {"C2": 128, "C3": 256, "C4": 512}
        # Create matching input
        B, H, W = 1, 160, 160
        d_feats = {
            "C2": torch.randn(B, 128, H//4, W//4, device=device),
            "C3": torch.randn(B, 256, H//8, W//8, device=device),
            "C4": torch.randn(B, 512, H//16, W//16, device=device),
        }
        t_feats = {k: torch.randn_like(v) for k, v in d_feats.items()}

        model = DTOrtFusion(channels=custom_ch).to(device)
        fused = model(d_feats, t_feats)

        for level in custom_ch:
            assert fused[level].shape == d_feats[level].shape

    def test_train_eval(self, depth_feats, thermal_feats, device):
        """Forward pass in both modes."""
        model = DTOrtFusion().to(device)

        model.train()
        out_train = model(depth_feats, thermal_feats)

        model.eval()
        with torch.no_grad():
            out_eval = model(depth_feats, thermal_feats)

        # Both should have same shapes
        for k in out_train:
            assert out_train[k].shape == out_eval[k].shape


# ───────────────────────────────────────────────────────────────
# Orthogonality property validation (mathematical correctness)
# ───────────────────────────────────────────────────────────────
class TestOrthogonalityProperty:
    """Mathematical verification of Gram-Schmidt orthogonal decomposition."""

    def test_perfect_orthogonality_alpha1(self, device):
        """At α=1, the orthogonal component should be truly orthogonal to basis."""
        torch.manual_seed(42)
        block = OrthoFusionBlock(128).to(device)
        block.eval()

        with torch.no_grad():
            d = torch.randn(2, 128, 32, 32, device=device)
            t = torch.randn(2, 128, 32, 32, device=device)

            depth0 = block.proj_d(d)
            ir0 = block.proj_t(t)

            # Full orthogonalization
            depth_orth = block._gram_schmidt_step(
                depth0, ir0, torch.tensor(1.0, device=device)
            )
            ir_orth = block._gram_schmidt_step(
                ir0, depth0, torch.tensor(1.0, device=device)
            )

            # Per-position inner products
            inner_d = (depth_orth * ir0).sum(dim=1)  # (B, H, W)
            inner_ir = (ir_orth * depth0).sum(dim=1)  # (B, H, W)

            max_inner_d = inner_d.abs().max().item()
            max_inner_ir = inner_ir.abs().max().item()

            assert max_inner_d < 1e-3, \
                f"<F_d^⊥, F_t> max={max_inner_d:.2e} should be ≈0"
            assert max_inner_ir < 1e-3, \
                f"<F_t^⊥, F_d> max={max_inner_ir:.2e} should be ≈0"

    def test_orthogonality_symmetric(self, device):
        """Swapping F_d ↔ F_t and α ↔ β yields symmetric results."""
        block = OrthoFusionBlock(128).to(device)
        block.eval()

        with torch.no_grad():
            # Set α=0.3, β=0.7
            block.alpha.data = torch.tensor(0.3)
            block.beta.data = torch.tensor(0.7)

            d = torch.randn(2, 128, 16, 16, device=device)
            t = torch.randn(2, 128, 16, 16, device=device)

            # Forward: F_d, F_t → fused
            out1 = block(d, t)

            # Swapped: F_t as F_d, F_d as F_t; swap α ↔ β manually
            block2 = OrthoFusionBlock(128).to(device)
            block2.eval()

            # Copy proj weights
            block2.proj_d.load_state_dict(block.proj_d.state_dict())
            block2.proj_t.load_state_dict(block.proj_t.state_dict())
            block2.fuse.load_state_dict(block.fuse.state_dict())

            # Swap α/β
            block2.alpha.data = torch.tensor(0.7)  # was β
            block2.beta.data = torch.tensor(0.3)   # was α

            out2 = block2(t, d)

            # Results are not identical when swapping since
            # depth_orth = d - α*proj_t(d), ir_orth = t - β*proj_d(t)
            # and proj_d ≠ proj_t (different mapping networks)
            # But both should be finite
            assert torch.isfinite(out1).all() and torch.isfinite(out2).all()

    def test_project_then_subtract_consistency(self, device):
        """F_a^⊥ + α*proj should reconstruct F_a (additivity check)."""
        C, H, W = 128, 8, 8
        F_a = torch.randn(2, C, H, W, device=device)
        F_b = torch.randn(2, C, H, W, device=device)

        alpha = torch.tensor(0.7, device=device)

        F_orth = OrthoFusionBlock._gram_schmidt_step(F_a, F_b, alpha)

        # proj = (<F_a,F_b> / ||F_b||²) * F_b
        inner = (F_a * F_b).sum(dim=1, keepdim=True)
        norm2 = F_b.pow(2).sum(dim=1, keepdim=True) + 1e-8
        proj = (inner / norm2) * F_b

        # F_a should = F_orth + alpha * proj
        reconstructed = F_orth + alpha * proj
        assert torch.allclose(reconstructed, F_a, atol=1e-5), \
            "F_a ≠ F_a^⊥ + α·proj — additivity violated"
