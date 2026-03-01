"""Export ACT's ResNet18 backbone (up to layer4) to ONNX.

Produces an ONNX model that takes (1, 3, H, W) input and returns
the layer4 feature map (1, 512, H/32, W/32), matching the exact
backbone used in ACT's modeling_act.py.

IMPORTANT: You must provide --policy-path to load the fine-tuned backbone
weights from the ACT checkpoint. Using fresh ImageNet weights will produce
incorrect feature maps at inference time.

Usage:
    python scripts/hailo/export_resnet18_onnx.py \
        --policy-path outputs/migrated/act_aloha_sim_transfer_cube_human \
        --output scripts/hailo/artifacts/resnet18_layer4.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision.models
from safetensors.torch import load_file
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


class ResNet18Layer4(torch.nn.Module):
    """Wrapper that returns just the layer4 feature map tensor (not a dict).

    IntermediateLayerGetter returns {"feature_map": tensor}, but ONNX export
    works more reliably with a single tensor output.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)["feature_map"]


def build_act_backbone(policy_path: str | None = None) -> IntermediateLayerGetter:
    """Build the ResNet18 backbone with fine-tuned weights from an ACT checkpoint.

    Args:
        policy_path: Path to a migrated ACT checkpoint directory containing
            model.safetensors. If None, uses fresh ImageNet weights (NOT
            recommended -- will produce wrong feature maps).

    Returns:
        IntermediateLayerGetter wrapping the ResNet18 up to layer4.
    """
    backbone_model = torchvision.models.resnet18(
        replace_stride_with_dilation=[False, False, False],
        weights="ResNet18_Weights.IMAGENET1K_V1",
        norm_layer=FrozenBatchNorm2d,
    )
    backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

    if policy_path:
        safetensors_path = Path(policy_path) / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {safetensors_path}")

        checkpoint = load_file(str(safetensors_path))

        # Extract backbone weights: "model.backbone.X.Y" -> "X.Y"
        backbone_state = {}
        for key, value in checkpoint.items():
            if key.startswith("model.backbone."):
                backbone_key = key[len("model.backbone.") :]
                backbone_state[backbone_key] = value

        n_loaded = len(backbone_state)
        backbone.load_state_dict(backbone_state, strict=True)
        print(f"Loaded {n_loaded} fine-tuned backbone parameters from {safetensors_path}")
    else:
        print("WARNING: No policy-path provided, using fresh ImageNet weights.")
        print("  The exported ONNX will NOT match the trained ACT policy!")

    backbone.eval()
    return backbone


def export_onnx(backbone, height: int, width: int, output_path: str):
    """Export the backbone to ONNX format using the legacy TorchScript exporter."""
    wrapper = ResNet18Layer4(backbone)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, height, width)

    with torch.no_grad():
        pt_output = wrapper(dummy_input)
    print(f"PyTorch output shape: {pt_output.shape}")

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["feature_map"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported ONNX model to {output_path}")

    # Validate the ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed")

    return pt_output


def verify_onnx(backbone, output_path: str, height: int, width: int):
    """Verify ONNX model output matches PyTorch output."""
    dummy_input = torch.randn(1, 3, height, width)

    # Run PyTorch
    with torch.no_grad():
        pt_out = backbone(dummy_input)["feature_map"]

    # Run ONNX inference
    session = ort.InferenceSession(output_path)
    onnx_out = session.run(None, {"input": dummy_input.numpy()})[0]

    # Compare
    max_diff = np.abs(pt_out.numpy() - onnx_out).max()
    mean_diff = np.abs(pt_out.numpy() - onnx_out).mean()
    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")

    if max_diff < 1e-4:
        print("ONNX verification PASSED")
    else:
        print(f"WARNING: max diff {max_diff:.8f} exceeds 1e-4 threshold")


def main():
    parser = argparse.ArgumentParser(
        description="Export ACT ResNet18 backbone to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/hailo/export_resnet18_onnx.py \\\n"
            "      --policy-path outputs/migrated/act_aloha_sim_transfer_cube_human \\\n"
            "      --output scripts/hailo/artifacts/resnet18_layer4.onnx\n"
        ),
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="outputs/migrated/act_aloha_sim_transfer_cube_human",
        help="Path to migrated ACT checkpoint (contains model.safetensors with fine-tuned backbone)",
    )
    parser.add_argument("--height", type=int, default=480, help="Input image height")
    parser.add_argument("--width", type=int, default=640, help="Input image width")
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/hailo/artifacts/resnet18_layer4.onnx",
        help="Output ONNX file path",
    )
    args = parser.parse_args()

    backbone = build_act_backbone(args.policy_path)
    export_onnx(backbone, args.height, args.width, args.output)
    verify_onnx(backbone, args.output, args.height, args.width)


if __name__ == "__main__":
    main()
