"""Compile ResNet18 ONNX model to Hailo HEF format.

Takes the ONNX model from export_resnet18_onnx.py and compiles it through
the Hailo SDK pipeline: parse -> optimize (INT8 quantization) -> compile.

Supports configurable optimization experiments via CLI flags for:
- Optimization level (0-4, with 4=AdaRound)
- Compression level (0-2)
- Activation/weights clipping on specified layers
- Per-layer FP16 precision (a16_w16)
- Post-quantization finetuning

Prerequisites:
    1. Export ONNX:  python scripts/hailo/export_resnet18_onnx.py
    2. Collect calibration images:
       MUJOCO_GL=egl python scripts/hailo/collect_calibration_images.py

Usage:
    # Basic (opt_level=4 with AdaRound)
    python scripts/hailo/compile_resnet18_hef.py \
        --calib-data scripts/hailo/artifacts/calibration_images.npy

    # With FP16 on layer4
    python scripts/hailo/compile_resnet18_hef.py \
        --fp16-layers "resnet18_layer4/layer4*" \
        --experiment-name fp16_layer4

    # Full experiment (clipping + FP16 + finetune)
    python scripts/hailo/compile_resnet18_hef.py \
        --optimization-level 4 \
        --activation-clipping "resnet18_layer4/layer3*,resnet18_layer4/layer4*" \
        --fp16-layers "resnet18_layer4/layer4*" \
        --finetune --finetune-epochs 8 --finetune-lr 0.0001 \
        --experiment-name full_opt

Requires: hailo_sdk_client (from hailo_dataflow_compiler wheel)

Outputs (in scripts/hailo/artifacts/):
    resnet18_layer4_<experiment>_parsed.har    -- HAR after parsing (FP32)
    resnet18_layer4_<experiment>_quantized.har -- HAR after INT8 quantization
    resnet18_layer4_<experiment>.hef           -- compiled HEF for device
"""

import argparse
from pathlib import Path

import numpy as np
from hailo_sdk_client import ClientRunner


def parse_onnx(onnx_path: str, hw_arch: str = "hailo10h", har_path: str | None = None) -> ClientRunner:
    """Parse ONNX model into Hailo format and save parsed HAR.

    Args:
        onnx_path: Path to the ONNX model file.
        hw_arch: Hailo hardware architecture target.
        har_path: Path to save the parsed HAR file. If None, not saved.

    Returns:
        Configured ClientRunner with the parsed model.
    """
    runner = ClientRunner(hw_arch=hw_arch)
    runner.translate_onnx_model(
        model=onnx_path,
        net_name="resnet18_layer4",
        net_input_shapes={"input": [1, 3, 480, 640]},
    )
    print("ONNX model parsed successfully")

    if har_path:
        runner.save_har(har_path)
        print(f"Parsed HAR saved to {har_path}")

    return runner


def build_model_script(
    optimization_level: int = 4,
    compression_level: int = 0,
    batch_size: int = 4,
    activation_clipping: str | None = None,
    weights_clipping: str | None = None,
    fp16_layers: str | None = None,
    finetune: bool = False,
    finetune_epochs: int = 8,
    finetune_lr: float = 0.0001,
    finetune_dataset_size: int = 4096,
    bias_correction: bool = True,
) -> str:
    """Build a Hailo model script from experiment parameters.

    Args:
        optimization_level: DFC optimization level (0-4). 4=AdaRound.
        compression_level: Weight compression (0=full precision, 2=most compressed).
        batch_size: Batch size for calibration (reduce for large inputs).
        activation_clipping: Comma-separated layer name patterns for activation clipping.
        weights_clipping: Comma-separated layer name patterns for weights clipping.
        fp16_layers: Comma-separated layer name patterns to keep in a16_w16 precision.
        finetune: Enable post-quantization finetuning.
        finetune_epochs: Number of finetuning epochs.
        finetune_lr: Finetuning learning rate.
        finetune_dataset_size: Number of samples for finetuning.
        bias_correction: Enable bias correction.

    Returns:
        Model script string for runner.load_model_script().
    """
    lines = []

    # Core optimization flavor
    lines.append(
        f"model_optimization_flavor("
        f"optimization_level={optimization_level}, "
        f"compression_level={compression_level}, "
        f"batch_size={batch_size})"
    )

    # Pre-quantization: activation clipping
    if activation_clipping:
        layers = [s.strip() for s in activation_clipping.split(",")]
        layers_str = ", ".join(layers)
        lines.append(
            f"pre_quantization_optimization(activation_clipping, "
            f"layers={{{layers_str}}}, mode=percentile, clipping_values=[0.01, 99.99])"
        )

    # Pre-quantization: weights clipping
    if weights_clipping:
        layers = [s.strip() for s in weights_clipping.split(",")]
        layers_str = ", ".join(layers)
        lines.append(
            f"pre_quantization_optimization(weights_clipping, "
            f"layers=[{layers_str}], mode=percentile, clipping_values=[0.01, 99.99])"
        )

    # Per-layer FP16 precision
    if fp16_layers:
        layers = [s.strip() for s in fp16_layers.split(",")]
        for layer in layers:
            lines.append(f"quantization_param({layer}, precision_mode=a16_w16)")

    # Post-quantization: bias correction
    if bias_correction:
        lines.append("post_quantization_optimization(bias_correction, policy=enabled)")

    # Post-quantization: finetuning
    if finetune:
        lines.append(
            f"post_quantization_optimization(finetune, policy=enabled, "
            f"learning_rate={finetune_lr}, epochs={finetune_epochs}, "
            f"dataset_size={finetune_dataset_size})"
        )

    script = "\n".join(lines)
    return script


def optimize(
    runner: ClientRunner,
    calib_path: str,
    model_script: str,
    har_path: str | None = None,
) -> ClientRunner:
    """Quantize model with given model script and calibration data.

    Args:
        runner: Parsed ClientRunner.
        calib_path: Path to .npy file with calibration images (NHWC, float32).
        model_script: Hailo model script string.
        har_path: Path to save the quantized HAR file. If None, not saved.

    Returns:
        ClientRunner with quantized model.
    """
    if not Path(calib_path).exists():
        raise FileNotFoundError(
            f"Calibration data not found: {calib_path}\n"
            "Generate it first:\n"
            "  MUJOCO_GL=egl python scripts/hailo/collect_calibration_images.py \\\n"
            "      --policy-path outputs/migrated/act_aloha_sim_transfer_cube_human \\\n"
            "      --output scripts/hailo/artifacts/calibration_images.npy"
        )

    calib_data = np.load(calib_path)
    print(f"Loaded calibration data from {calib_path}:")
    print(f"  shape={calib_data.shape}, dtype={calib_data.dtype}")
    print(f"  range=[{calib_data.min():.3f}, {calib_data.max():.3f}]")
    print(f"  mean={calib_data.mean():.3f}, std={calib_data.std():.3f}")

    print("\nModel script:")
    for line in model_script.strip().split("\n"):
        print(f"  {line}")

    runner.load_model_script(model_script)

    print(f"\nRunning INT8 quantization with {calib_data.shape[0]} calibration samples...")
    runner.optimize(calib_data)
    print("Optimization (INT8 quantization) complete")

    if har_path:
        runner.save_har(har_path)
        print(f"Quantized HAR saved to {har_path}")

    return runner


def compile_hef(runner: ClientRunner, output_path: str):
    """Compile the quantized model to HEF."""
    print("Compiling to HEF...")
    hef_data = runner.compile()

    with open(output_path, "wb") as f:
        f.write(hef_data)
    print(f"HEF saved to {output_path} ({len(hef_data)} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Compile ResNet18 ONNX to Hailo HEF with configurable optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Baseline with AdaRound (opt_level=4)\n"
            "  python scripts/hailo/compile_resnet18_hef.py --experiment-name opt4_baseline\n"
            "\n"
            "  # FP16 on layer4\n"
            "  python scripts/hailo/compile_resnet18_hef.py \\\n"
            "      --fp16-layers 'resnet18_layer4/layer4*' \\\n"
            "      --experiment-name fp16_layer4\n"
            "\n"
            "  # Full: clipping + FP16 + finetune\n"
            "  python scripts/hailo/compile_resnet18_hef.py \\\n"
            "      --activation-clipping 'resnet18_layer4/layer3*,resnet18_layer4/layer4*' \\\n"
            "      --fp16-layers 'resnet18_layer4/layer4*' \\\n"
            "      --finetune --experiment-name full_opt\n"
        ),
    )

    # Input / output
    parser.add_argument(
        "--onnx",
        type=str,
        default="scripts/hailo/artifacts/resnet18_layer4.onnx",
        help="Input ONNX file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/hailo/artifacts",
        help="Directory for output artifacts",
    )
    parser.add_argument("--hw-arch", type=str, default="hailo10h", help="Hailo hardware architecture")
    parser.add_argument(
        "--calib-data",
        type=str,
        default="scripts/hailo/artifacts/calibration_images.npy",
        help="Path to .npy calibration images",
    )

    # Experiment identification
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Tag for output artifacts (e.g. 'opt4_baseline'). Defaults to opt<level>.",
    )

    # Optimization parameters
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=4,
        choices=[0, 1, 2, 3, 4],
        help="DFC optimization level: 0=equalization, 1=+IBC, 2=+finetune, 3=+AdaRound(256), 4=+AdaRound(1024)",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Weight compression: 0=full precision (default), 2=most compressed",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Calibration batch size")

    # Clipping
    parser.add_argument(
        "--activation-clipping",
        type=str,
        default=None,
        help="Comma-separated layer patterns for activation clipping (e.g. 'resnet18_layer4/layer4*')",
    )
    parser.add_argument(
        "--weights-clipping",
        type=str,
        default=None,
        help="Comma-separated layer patterns for weights clipping",
    )

    # Per-layer precision
    parser.add_argument(
        "--fp16-layers",
        type=str,
        default=None,
        help="Comma-separated layer patterns to keep in a16_w16 FP16 precision",
    )

    # Post-quantization
    parser.add_argument("--no-bias-correction", action="store_true", help="Disable bias correction")
    parser.add_argument("--finetune", action="store_true", help="Enable post-quantization finetuning")
    parser.add_argument("--finetune-epochs", type=int, default=8, help="Finetuning epochs")
    parser.add_argument("--finetune-lr", type=float, default=0.0001, help="Finetuning learning rate")
    parser.add_argument(
        "--finetune-dataset-size", type=int, default=4096, help="Number of samples for finetuning"
    )

    # Control
    parser.add_argument("--parse-only", action="store_true", help="Only parse ONNX (skip optimize + compile)")
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Parse + optimize but skip compile (useful for noise analysis)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print model script without running")

    args = parser.parse_args()

    # Determine experiment name
    experiment = args.experiment_name or f"opt{args.optimization_level}"

    # Build output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"resnet18_layer4_{experiment}"
    parsed_har = str(output_dir / f"{base}_parsed.har")
    quantized_har = str(output_dir / f"{base}_quantized.har")
    hef_path = str(output_dir / f"{base}.hef")

    # Build model script
    model_script = build_model_script(
        optimization_level=args.optimization_level,
        compression_level=args.compression_level,
        batch_size=args.batch_size,
        activation_clipping=args.activation_clipping,
        weights_clipping=args.weights_clipping,
        fp16_layers=args.fp16_layers,
        finetune=args.finetune,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        finetune_dataset_size=args.finetune_dataset_size,
        bias_correction=not args.no_bias_correction,
    )

    print(f"Experiment: {experiment}")
    print(f"Model script:\n{model_script}\n")

    if args.dry_run:
        print("Dry run — exiting without running.")
        print("\nWould produce:")
        print(f"  Parsed HAR:    {parsed_har}")
        print(f"  Quantized HAR: {quantized_har}")
        print(f"  HEF:           {hef_path}")
        return

    if not Path(args.onnx).exists():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}. Run export_resnet18_onnx.py first.")

    # Parse
    runner = parse_onnx(args.onnx, args.hw_arch, har_path=parsed_har)

    if args.parse_only:
        print("\nParse-only mode — stopping after parse.")
        print(f"  Parsed HAR: {parsed_har}")
        return

    # Optimize
    runner = optimize(runner, calib_path=args.calib_data, model_script=model_script, har_path=quantized_har)

    if args.skip_compile:
        print("\nSkip-compile mode — stopping after optimize.")
        print(f"  Parsed HAR:    {parsed_har}")
        print(f"  Quantized HAR: {quantized_har}")
        return

    # Compile
    compile_hef(runner, hef_path)

    print("\nArtifacts:")
    print(f"  Parsed HAR:    {parsed_har}")
    print(f"  Quantized HAR: {quantized_har}")
    print(f"  HEF:           {hef_path}")


if __name__ == "__main__":
    main()
