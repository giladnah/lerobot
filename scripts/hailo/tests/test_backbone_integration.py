"""Verify HailoBackbone drop-in module works correctly.

Tests that HailoBackbone produces the right output shape, format, and dtype
when given standard ACT image inputs.

Usage:
    pytest scripts/hailo/tests/test_backbone_integration.py -v \
        --hef scripts/hailo/artifacts/resnet18_layer4.hef

Requires: hailo_platform
"""

from pathlib import Path

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--hef",
        default="scripts/hailo/artifacts/resnet18_layer4.hef",
        help="Path to compiled HEF file",
    )


@pytest.fixture
def hef_path(request):
    path = request.config.getoption("--hef")
    if not Path(path).exists():
        pytest.skip(f"HEF not found: {path}")
    return path


@pytest.fixture
def backbone(hef_path):
    from lerobot.policies.act.hailo_backbone import HailoBackbone

    bb = HailoBackbone(hef_path)
    yield bb
    bb.close()


class TestBackboneIntegration:
    def test_output_shape_single(self, backbone):
        """Verify output shape for single image (B=1)."""
        x = torch.randn(1, 3, 480, 640)
        out = backbone(x)
        assert "feature_map" in out
        assert out["feature_map"].shape == (1, 512, 15, 20)

    def test_output_shape_batch(self, backbone):
        """Verify output shape for batched input (B=2)."""
        x = torch.randn(2, 3, 480, 640)
        out = backbone(x)
        assert out["feature_map"].shape == (2, 512, 15, 20)

    def test_output_dtype(self, backbone):
        """Verify output is float32."""
        x = torch.randn(1, 3, 480, 640)
        out = backbone(x)
        assert out["feature_map"].dtype == torch.float32

    def test_output_dict_format(self, backbone):
        """Verify output is a dict with 'feature_map' key (matching IntermediateLayerGetter)."""
        x = torch.randn(1, 3, 480, 640)
        out = backbone(x)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"feature_map"}

    def test_deterministic_output(self, backbone):
        """Verify same input produces same output."""
        x = torch.randn(1, 3, 480, 640)
        out1 = backbone(x)["feature_map"]
        out2 = backbone(x)["feature_map"]
        assert torch.allclose(out1, out2, atol=1e-6), "Hailo backbone is not deterministic"

    def test_output_on_device(self, backbone):
        """Verify output tensor is on the same device as input."""
        x = torch.randn(1, 3, 480, 640)
        out = backbone(x)
        assert out["feature_map"].device == x.device
