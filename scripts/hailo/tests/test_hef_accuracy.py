"""Verify HEF (INT8) accuracy against ONNX (FP32) reference.

Runs calibration images through both ONNX Runtime (FP32) and the Hailo HEF
(INT8) and reports SNR, cosine similarity, and max absolute difference.

Usage:
    pytest scripts/hailo/tests/test_hef_accuracy.py -v \
        --hef scripts/hailo/artifacts/resnet18_layer4.hef \
        --onnx scripts/hailo/artifacts/resnet18_layer4.onnx \
        --calib-data scripts/hailo/artifacts/calibration_images.npy

Requires: hailo_platform, onnxruntime
"""

from pathlib import Path

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--hef",
        default="scripts/hailo/artifacts/resnet18_layer4.hef",
        help="Path to compiled HEF file",
    )
    parser.addoption(
        "--onnx",
        default="scripts/hailo/artifacts/resnet18_layer4.onnx",
        help="Path to ONNX model (FP32 reference)",
    )
    parser.addoption(
        "--calib-data",
        default="scripts/hailo/artifacts/calibration_images.npy",
        help="Path to calibration images (NHWC float32)",
    )
    parser.addoption(
        "--n-samples",
        default=10,
        type=int,
        help="Number of calibration samples to test",
    )


@pytest.fixture
def hef_path(request):
    path = request.config.getoption("--hef")
    if not Path(path).exists():
        pytest.skip(f"HEF not found: {path}")
    return path


@pytest.fixture
def onnx_path(request):
    path = request.config.getoption("--onnx")
    if not Path(path).exists():
        pytest.skip(f"ONNX not found: {path}")
    return path


@pytest.fixture
def calib_data(request):
    path = request.config.getoption("--calib-data")
    if not Path(path).exists():
        pytest.skip(f"Calibration data not found: {path}")
    data = np.load(path)
    n = request.config.getoption("--n-samples")
    return data[:n]


@pytest.fixture
def onnx_session(onnx_path):
    import onnxruntime as ort

    return ort.InferenceSession(onnx_path)


@pytest.fixture
def hailo_model(hef_path):
    from hailo_platform import FormatType, VDevice

    vdevice = VDevice()
    infer_model = vdevice.create_infer_model(hef_path)
    infer_model.input().set_format_type(FormatType.FLOAT32)
    infer_model.output().set_format_type(FormatType.FLOAT32)
    configured = infer_model.configure()
    yield configured, infer_model.output().shape
    vdevice.release()


def run_onnx_inference(session, image_nhwc: np.ndarray) -> np.ndarray:
    """Run ONNX FP32 inference on a single NHWC image."""
    # ONNX expects NCHW
    image_nchw = np.transpose(image_nhwc, (2, 0, 1))[np.newaxis]  # (1, 3, H, W)
    output = session.run(None, {"input": image_nchw.astype(np.float32)})[0]
    return output.squeeze(0)  # (C, H', W')


def run_hailo_inference(configured_model, output_shape, image_nhwc: np.ndarray) -> np.ndarray:
    """Run Hailo INT8 inference on a single NHWC image."""
    bindings = configured_model.create_bindings()
    sample = np.ascontiguousarray(image_nhwc, dtype=np.float32)
    bindings.input().set_buffer(sample)
    out_buffer = np.empty(output_shape, dtype=np.float32)
    bindings.output().set_buffer(out_buffer)
    configured_model.run([bindings], timeout=10000)
    output = bindings.output().get_buffer()
    # Convert NHWC -> CHW
    return np.transpose(output, (2, 0, 1))  # (C, H', W')


class TestHefAccuracy:
    def test_snr_above_threshold(self, onnx_session, hailo_model, calib_data):
        """Verify overall SNR is above 20 dB."""
        configured, out_shape = hailo_model
        snrs = []

        for i in range(len(calib_data)):
            fp32 = run_onnx_inference(onnx_session, calib_data[i])
            int8 = run_hailo_inference(configured, out_shape, calib_data[i])
            noise = fp32 - int8
            signal_power = np.mean(fp32**2)
            noise_power = np.mean(noise**2)
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                snrs.append(snr)

        mean_snr = np.mean(snrs)
        print(f"\nMean SNR across {len(snrs)} samples: {mean_snr:.2f} dB")
        assert mean_snr > 20.0, f"Mean SNR {mean_snr:.2f} dB is below 20 dB threshold"

    def test_cosine_similarity(self, onnx_session, hailo_model, calib_data):
        """Verify mean cosine similarity is above 0.99."""
        configured, out_shape = hailo_model
        cosines = []

        for i in range(len(calib_data)):
            fp32 = run_onnx_inference(onnx_session, calib_data[i]).flatten()
            int8 = run_hailo_inference(configured, out_shape, calib_data[i]).flatten()
            cos = np.dot(fp32, int8) / (np.linalg.norm(fp32) * np.linalg.norm(int8) + 1e-8)
            cosines.append(cos)

        mean_cos = np.mean(cosines)
        print(f"\nMean cosine similarity: {mean_cos:.6f}")
        assert mean_cos > 0.99, f"Mean cosine {mean_cos:.6f} is below 0.99 threshold"

    def test_output_shape(self, hailo_model, calib_data):
        """Verify HEF output shape matches expected dimensions."""
        configured, out_shape = hailo_model
        output = run_hailo_inference(configured, out_shape, calib_data[0])
        # Expected: (512, 15, 20) for 480x640 input
        assert output.shape == (512, 15, 20), f"Unexpected output shape: {output.shape}"

    def test_feature_statistics(self, onnx_session, hailo_model, calib_data):
        """Report per-sample statistics (informational, no hard assertions)."""
        configured, out_shape = hailo_model

        all_max_diff = []
        all_mean_diff = []

        for i in range(len(calib_data)):
            fp32 = run_onnx_inference(onnx_session, calib_data[i])
            int8 = run_hailo_inference(configured, out_shape, calib_data[i])
            diff = np.abs(fp32 - int8)
            all_max_diff.append(diff.max())
            all_mean_diff.append(diff.mean())

        print(f"\nMax abs diff: mean={np.mean(all_max_diff):.4f}, max={np.max(all_max_diff):.4f}")
        print(f"Mean abs diff: mean={np.mean(all_mean_diff):.4f}")
