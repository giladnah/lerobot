"""Analyze per-layer quantization noise in the ResNet18 Hailo model.

Uses the Hailo DFC's noise analysis to compute per-layer SNR (signal-to-noise
ratio) for the INT8 quantized model. Layers with low SNR are candidates for
activation/weights clipping or FP16 precision to improve accuracy.

Prerequisites:
    1. Parsed HAR:  python scripts/hailo/compile_resnet18_hef.py (or just parse step)
    2. Calibration data:  MUJOCO_GL=egl python scripts/hailo/collect_calibration_images.py

Usage:
    python scripts/hailo/analyze_layer_noise.py \
        --har scripts/hailo/artifacts/resnet18_layer4_quantized.har \
        --calib-data scripts/hailo/artifacts/calibration_images.npy \
        --data-count 1024

Requires: hailo_sdk_client (from hailo_dataflow_compiler wheel)

Outputs:
    Per-layer SNR table (printed to stdout)
    Optionally saved to scripts/hailo/reports/report_layer_noise_analysis.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
from hailo_sdk_client import ClientRunner


def analyze_noise(har_path: str, calib_path: str, data_count: int = 1024) -> dict:
    """Run noise analysis on a quantized HAR model.

    Args:
        har_path: Path to the quantized HAR file.
        calib_path: Path to .npy calibration data (NHWC float32).
        data_count: Number of calibration samples to use.

    Returns:
        Dict with per-layer noise analysis results.
    """
    if not Path(har_path).exists():
        raise FileNotFoundError(f"HAR file not found: {har_path}")
    if not Path(calib_path).exists():
        raise FileNotFoundError(
            f"Calibration data not found: {calib_path}\n"
            "Generate it first:\n"
            "  MUJOCO_GL=egl python scripts/hailo/collect_calibration_images.py"
        )

    calib_data = np.load(calib_path)
    print(f"Loaded calibration data: shape={calib_data.shape}, dtype={calib_data.dtype}")
    print(f"Using {min(data_count, calib_data.shape[0])} samples for noise analysis")

    runner = ClientRunner(har=har_path)
    print(f"Loaded HAR from {har_path}")

    print("Running noise analysis (this may take a few minutes)...")
    noise_results = runner.analyze_noise(calib_data, data_count=data_count)

    return noise_results


def format_results(noise_results, snr_threshold: float = 25.0) -> str:
    """Format noise analysis results as a readable table.

    Args:
        noise_results: Results from runner.analyze_noise().
        snr_threshold: SNR below this value is flagged as low.

    Returns:
        Formatted string with per-layer SNR table.
    """
    lines = []
    lines.append(f"{'Layer':<50s}  {'SNR (dB)':>10s}  {'Status':>10s}")
    lines.append("-" * 75)

    low_snr_layers = []

    if isinstance(noise_results, dict):
        for layer_name, metrics in sorted(noise_results.items()):
            if isinstance(metrics, dict):
                snr = metrics.get("snr", metrics.get("SNR", None))
            else:
                snr = float(metrics)

            if snr is not None:
                status = "LOW" if snr < snr_threshold else "OK"
                lines.append(f"{layer_name:<50s}  {snr:>10.2f}  {status:>10s}")
                if snr < snr_threshold:
                    low_snr_layers.append((layer_name, snr))
    else:
        # If noise_results is not a dict, print it as-is
        lines.append(f"Raw results: {noise_results}")

    lines.append("")
    lines.append(f"SNR threshold: {snr_threshold:.1f} dB")
    lines.append(f"Layers below threshold: {len(low_snr_layers)}")

    if low_snr_layers:
        lines.append("\nLow-SNR layers (candidates for FP16 or clipping):")
        for name, snr in sorted(low_snr_layers, key=lambda x: x[1]):
            lines.append(f"  {name}: {snr:.2f} dB")

    return "\n".join(lines)


def save_report(noise_results, output_path: str, har_path: str, snr_threshold: float = 25.0):
    """Save noise analysis results as a markdown report."""
    report = []
    report.append("# Layer Noise Analysis Report\n")
    report.append(f"**HAR**: `{har_path}`\n")
    report.append(f"**SNR Threshold**: {snr_threshold:.1f} dB\n")
    report.append("## Per-Layer SNR\n")
    report.append(f"| {'Layer':<50s} | {'SNR (dB)':>10s} | {'Status':>10s} |")
    report.append(f"|{'-' * 52}|{'-' * 12}|{'-' * 12}|")

    low_snr_layers = []

    if isinstance(noise_results, dict):
        for layer_name, metrics in sorted(noise_results.items()):
            if isinstance(metrics, dict):
                snr = metrics.get("snr", metrics.get("SNR", None))
            else:
                snr = float(metrics)

            if snr is not None:
                status = "**LOW**" if snr < snr_threshold else "OK"
                report.append(f"| {layer_name:<50s} | {snr:>10.2f} | {status:>10s} |")
                if snr < snr_threshold:
                    low_snr_layers.append((layer_name, snr))

    report.append("")
    report.append("## Summary\n")
    report.append(
        f"- Total layers analyzed: {len(noise_results) if isinstance(noise_results, dict) else 'N/A'}"
    )
    report.append(f"- Layers below threshold: {len(low_snr_layers)}")

    if low_snr_layers:
        report.append("\n## Recommended Actions\n")
        report.append("These layers have low SNR and are candidates for:\n")
        report.append("1. **Activation/weights clipping**: Reduce outlier impact on quantization ranges")
        report.append("2. **FP16 precision** (`a16_w16`): Keep layer in 16-bit for higher accuracy")
        report.append("")
        for name, snr in sorted(low_snr_layers, key=lambda x: x[1]):
            report.append(f"- `{name}`: {snr:.2f} dB")

        report.append("\n### Model script snippet for FP16 on low-SNR layers\n")
        report.append("```")
        for name, _snr in sorted(low_snr_layers, key=lambda x: x[1]):
            report.append(f"quantization_param({name}, precision_mode=a16_w16)")
        report.append("```")

    report_text = "\n".join(report) + "\n"
    Path(output_path).write_text(report_text)
    print(f"\nReport saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-layer quantization noise in Hailo model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/hailo/analyze_layer_noise.py \\\n"
            "      --har scripts/hailo/artifacts/resnet18_layer4_quantized.har \\\n"
            "      --calib-data scripts/hailo/artifacts/calibration_images.npy\n"
        ),
    )
    parser.add_argument(
        "--har",
        type=str,
        default="scripts/hailo/artifacts/resnet18_layer4_quantized.har",
        help="Path to quantized HAR file",
    )
    parser.add_argument(
        "--calib-data",
        type=str,
        default="scripts/hailo/artifacts/calibration_images.npy",
        help="Path to .npy calibration data (NHWC float32)",
    )
    parser.add_argument("--data-count", type=int, default=1024, help="Number of calibration samples to use")
    parser.add_argument(
        "--snr-threshold", type=float, default=25.0, help="SNR below this value is flagged as low (dB)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/hailo/reports/report_layer_noise_analysis.md",
        help="Path to save markdown report",
    )
    parser.add_argument("--json", type=str, default=None, help="Path to save raw results as JSON")
    args = parser.parse_args()

    noise_results = analyze_noise(args.har, args.calib_data, args.data_count)

    # Print formatted results
    print("\n" + format_results(noise_results, args.snr_threshold))

    # Save markdown report
    save_report(noise_results, args.output, args.har, args.snr_threshold)

    # Optionally save raw JSON
    if args.json:
        with open(args.json, "w") as f:
            json.dump(noise_results, f, indent=2, default=str)
        print(f"Raw JSON saved to {args.json}")


if __name__ == "__main__":
    main()
