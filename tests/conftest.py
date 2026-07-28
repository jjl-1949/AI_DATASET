"""Shared fixtures for all test modules."""

import sys
import os
import pytest
import torch

# Ensure root dir is in path for imports
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Core fixtures ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def device() -> torch.device:
    """Use CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def batch_size() -> int:
    return 2


@pytest.fixture(scope="session")
def img_hw() -> tuple:
    """Standard input size H, W."""
    return (320, 320)


# ── Input tensor fixtures ──────────────────────────────────────

@pytest.fixture
def depth_input(batch_size, img_hw, device) -> torch.Tensor:
    """Single-channel depth input (B, 1, H, W)."""
    return torch.randn(batch_size, 1, *img_hw, device=device)


@pytest.fixture
def thermal_input(batch_size, img_hw, device) -> torch.Tensor:
    """Single-channel thermal input (B, 1, H, W)."""
    return torch.randn(batch_size, 1, *img_hw, device=device)


@pytest.fixture
def rgb_input(batch_size, img_hw, device) -> torch.Tensor:
    """3-channel RGB input (B, 3, H, W)."""
    return torch.randn(batch_size, 3, *img_hw, device=device)


# ── Multi-scale feature fixtures ───────────────────────────────

@pytest.fixture
def multiscale_feats(batch_size, img_hw, device) -> dict:
    """Multi-scale features in ResNet50 format (C2–C5)."""
    H, W = img_hw
    return {
        "C2": torch.randn(batch_size, 256,  H // 4,  W // 4,  device=device),
        "C3": torch.randn(batch_size, 512,  H // 8,  W // 8,  device=device),
        "C4": torch.randn(batch_size, 1024, H // 16, W // 16, device=device),
        "C5": torch.randn(batch_size, 2048, H // 32, W // 32, device=device),
    }


@pytest.fixture
def depth_feats(batch_size, img_hw, device) -> dict:
    """Depth ResNet50 outputs — same structure as multiscale_feats."""
    H, W = img_hw
    return {
        "C2": torch.randn(batch_size, 256,  H // 4,  W // 4,  device=device),
        "C3": torch.randn(batch_size, 512,  H // 8,  W // 8,  device=device),
        "C4": torch.randn(batch_size, 1024, H // 16, W // 16, device=device),
        "C5": torch.randn(batch_size, 2048, H // 32, W // 32, device=device),
    }


@pytest.fixture
def thermal_feats(batch_size, img_hw, device) -> dict:
    """Thermal ResNet50 outputs."""
    H, W = img_hw
    return {
        "C2": torch.randn(batch_size, 256,  H // 4,  W // 4,  device=device),
        "C3": torch.randn(batch_size, 512,  H // 8,  W // 8,  device=device),
        "C4": torch.randn(batch_size, 1024, H // 16, W // 16, device=device),
        "C5": torch.randn(batch_size, 2048, H // 32, W // 32, device=device),
    }
