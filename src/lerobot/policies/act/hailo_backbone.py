"""Hailo-accelerated ResNet18 backbone for ACT policy.

Drop-in replacement for IntermediateLayerGetter(ResNet18, {"layer4": "feature_map"}).
Runs ResNet18 (up to layer4) inference on a Hailo accelerator using a pre-compiled HEF file.

Returns the same output format: dict with "feature_map" key containing a
(B, 512, H/32, W/32) float32 tensor.

Usage:
    backbone = HailoBackbone("resnet18_layer4.hef")
    features = backbone(images)  # images: (B, 3, H, W) float32
    feature_map = features["feature_map"]  # (B, 512, H/32, W/32) float32
"""

import numpy as np
import torch
import torch.nn as nn
from hailo_platform import FormatType, VDevice


class HailoBackbone(nn.Module):
    """Hailo-accelerated ResNet18 backbone (up to layer4).

    This module offloads ResNet18 feature extraction to a Hailo accelerator.
    The HEF file must be pre-compiled from the same ResNet18 architecture
    used by ACT (see scripts/compile_resnet18_hef.py).

    The Hailo device processes one sample at a time. For batched inputs,
    inference is run sequentially over the batch dimension.
    """

    def __init__(self, hef_path: str):
        super().__init__()
        self.hef_path = hef_path

        # Initialize Hailo device and model
        self._vdevice = VDevice()
        infer_model = self._vdevice.create_infer_model(hef_path)

        # Request float32 I/O so HailoRT handles quantization/dequantization
        infer_model.input().set_format_type(FormatType.FLOAT32)
        infer_model.output().set_format_type(FormatType.FLOAT32)

        self._input_shape = infer_model.input().shape  # [H, W, C] NHWC
        self._output_shape = infer_model.output().shape  # [H', W', C'] NHWC

        self._configured_model = infer_model.configure()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run Hailo inference on input images.

        Args:
            x: Input tensor of shape (B, C, H, W), float32.

        Returns:
            Dict with "feature_map" key containing (B, 512, H/32, W/32) float32 tensor.
        """
        device = x.device
        batch_size = x.shape[0]

        # Convert from PyTorch NCHW to Hailo NHWC
        x_nhwc = x.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy()

        outputs = []
        for i in range(batch_size):
            sample = np.ascontiguousarray(x_nhwc[i])
            out = self._infer_single(sample)
            outputs.append(out)

        # Stack batch and convert from NHWC to NCHW
        output_nhwc = np.stack(outputs, axis=0)  # (B, H', W', C')
        output_tensor = torch.from_numpy(output_nhwc).permute(0, 3, 1, 2).to(device)

        return {"feature_map": output_tensor}

    def _infer_single(self, sample: np.ndarray) -> np.ndarray:
        """Run inference on a single sample.

        Args:
            sample: Input array of shape (H, W, C), float32, C-contiguous.

        Returns:
            Output array of shape (H', W', C'), float32.
        """
        bindings = self._configured_model.create_bindings()
        bindings.input().set_buffer(sample)
        out_buffer = np.empty(self._output_shape, dtype=np.float32)
        bindings.output().set_buffer(out_buffer)
        self._configured_model.run([bindings], timeout=10000)
        return bindings.output().get_buffer()

    def close(self):
        """Release the Hailo device."""
        if hasattr(self, "_vdevice") and self._vdevice is not None:
            self._vdevice.release()
            self._vdevice = None

    def __del__(self):
        self.close()
