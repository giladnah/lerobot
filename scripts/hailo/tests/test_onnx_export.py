"""Verify ONNX export matches PyTorch backbone output.

Loads the fine-tuned backbone from the ACT checkpoint, exports to ONNX,
and verifies that both produce identical outputs for the same input.

Usage:
    pytest scripts/hailo/tests/test_onnx_export.py -v \
        --policy-path outputs/migrated/act_aloha_sim_transfer_cube_human

Requires: onnxruntime, safetensors, torchvision
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add scripts/hailo to sys.path so we can import export_resnet18_onnx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from export_resnet18_onnx import ResNet18Layer4, build_act_backbone


def pytest_addoption(parser):
    parser.addoption(
        "--policy-path",
        default="outputs/migrated/act_aloha_sim_transfer_cube_human",
        help="Path to migrated ACT checkpoint",
    )


@pytest.fixture
def policy_path(request):
    return request.config.getoption("--policy-path")


@pytest.fixture
def backbone(policy_path):
    """Load the fine-tuned backbone from the ACT checkpoint."""
    path = Path(policy_path) / "model.safetensors"
    if not path.exists():
        pytest.skip(f"Checkpoint not found: {path}")
    return build_act_backbone(policy_path)


@pytest.fixture
def onnx_path(backbone, tmp_path):
    """Export backbone to ONNX and return the path."""
    import onnx

    wrapper = ResNet18Layer4(backbone)
    wrapper.eval()
    output_path = str(tmp_path / "resnet18_layer4.onnx")
    dummy = torch.randn(1, 3, 480, 640)
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["feature_map"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    return output_path


class TestOnnxExport:
    def test_onnx_matches_pytorch(self, backbone, onnx_path):
        """Verify ONNX output matches PyTorch within tight tolerance."""
        import onnxruntime as ort

        dummy = torch.randn(1, 3, 480, 640)

        # PyTorch reference
        with torch.no_grad():
            pt_out = backbone(dummy)["feature_map"].numpy()

        # ONNX inference
        session = ort.InferenceSession(onnx_path)
        onnx_out = session.run(None, {"input": dummy.numpy()})[0]

        max_diff = np.abs(pt_out - onnx_out).max()
        cosine = np.dot(pt_out.flatten(), onnx_out.flatten()) / (
            np.linalg.norm(pt_out.flatten()) * np.linalg.norm(onnx_out.flatten())
        )

        assert max_diff < 1e-4, f"Max abs diff {max_diff:.8f} exceeds 1e-4"
        assert cosine > 0.9999, f"Cosine similarity {cosine:.6f} below 0.9999"

    def test_output_shape(self, backbone):
        """Verify backbone produces correct output shape."""
        dummy = torch.randn(1, 3, 480, 640)
        with torch.no_grad():
            out = backbone(dummy)
        assert "feature_map" in out
        assert out["feature_map"].shape == (1, 512, 15, 20)

    def test_output_deterministic(self, backbone):
        """Verify backbone output is deterministic in eval mode."""
        dummy = torch.randn(1, 3, 480, 640)
        backbone.eval()
        with torch.no_grad():
            out1 = backbone(dummy)["feature_map"]
            out2 = backbone(dummy)["feature_map"]
        assert torch.allclose(out1, out2, atol=0), "Backbone is not deterministic"

    def test_batch_consistency(self, backbone):
        """Verify batched output matches single-sample outputs."""
        img1 = torch.randn(1, 3, 480, 640)
        img2 = torch.randn(1, 3, 480, 640)
        batch = torch.cat([img1, img2], dim=0)

        with torch.no_grad():
            out1 = backbone(img1)["feature_map"]
            out2 = backbone(img2)["feature_map"]
            out_batch = backbone(batch)["feature_map"]

        assert torch.allclose(out_batch[0], out1[0], atol=1e-6)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-6)
