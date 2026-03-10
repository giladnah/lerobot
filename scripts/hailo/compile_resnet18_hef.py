"""Compile ResNet18 ONNX model to Hailo HEF format.

Takes the ONNX model from export_resnet18_onnx.py and compiles it through
the Hailo SDK pipeline: parse -> optimize (INT8 quantization) -> compile.

Supports configurable optimization experiments via CLI flags for:
- Optimization level (0-4, with 4=AdaRound)
- Compression level (0-2)
- Activation/weights clipping on specified layers
- Per-layer precision mode (a16_w16, a16_w8, a16_w4, a8_w4)
- Calibration dataset size (calibset_size — DFC default is only 64)
- Post-quantization finetuning (with configurable loss type)

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
    calibset_size: int = 1024,
    normalization_mean: list[float] | None = None,
    normalization_std: list[float] | None = None,
    activation_clipping: str | None = None,
    weights_clipping: str | None = None,
    fp16_layers: str | None = None,
    precision_mode: str = "a16_w16",
    fp16_layers_b: str | None = None,
    precision_mode_b: str = "a16_w16",
    finetune: bool = False,
    finetune_epochs: int = 8,
    finetune_lr: float = 0.0001,
    finetune_dataset_size: int = 4096,
    finetune_loss_type: str | None = None,
    bias_correction: bool = True,
    quantization_groups: int | None = None,
    quantization_groups_layers: str | None = None,
) -> str:
    """Build a Hailo model script from experiment parameters.

    Args:
        optimization_level: DFC optimization level (0-4). 4=AdaRound.
        compression_level: Weight compression (0=full precision, 2=most compressed).
        batch_size: Batch size for calibration (reduce for large inputs).
        calibset_size: Number of images used for calibration statistics (DFC default=64).
        normalization_mean: Per-channel mean for input normalization (fused into graph).
        normalization_std: Per-channel std for input normalization (fused into graph).
        activation_clipping: Comma-separated layer name patterns for activation clipping.
        weights_clipping: Comma-separated layer name patterns for weights clipping.
        fp16_layers: Comma-separated layer name patterns for non-INT8 precision.
        precision_mode: Precision mode for fp16_layers (a16_w16, a16_w8, a16_w4, a8_w4).
        fp16_layers_b: Second set of layers with a different precision mode.
        precision_mode_b: Precision mode for fp16_layers_b.
        finetune: Enable post-quantization finetuning.
        finetune_epochs: Number of finetuning epochs.
        finetune_lr: Finetuning learning rate.
        finetune_dataset_size: Number of samples for finetuning.
        finetune_loss_type: Loss function for finetuning (l2, l2rel, cosine, ce). None=DFC default.
        bias_correction: Enable bias correction.
        quantization_groups: Split weights into N groups for independent quantization (2-4).
        quantization_groups_layers: Comma-separated layer patterns for quantization_groups.
            If None, applies to all conv layers ({conv*}).

    Returns:
        Model script string for runner.load_model_script().
    """
    lines = []

    # Input normalization — fused into first conv layer by the DFC.
    # Calibration data must be in pre-normalization range (e.g. [0, 255]).
    # Formula: normalized = (x - mean) / std
    if normalization_mean is not None and normalization_std is not None:
        mean_str = ", ".join(f"{v}" for v in normalization_mean)
        std_str = ", ".join(f"{v}" for v in normalization_std)
        lines.append(f"input_normalization1 = normalization([{mean_str}], [{std_str}])")

    # Core optimization flavor
    lines.append(
        f"model_optimization_flavor("
        f"optimization_level={optimization_level}, "
        f"compression_level={compression_level}, "
        f"batch_size={batch_size})"
    )

    # Calibration config — explicitly set calibset_size so all provided images
    # are used for calibration statistics (DFC default is only 64)
    lines.append(
        f"model_optimization_config(calibration, "
        f"batch_size={batch_size}, calibset_size={calibset_size})"
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

    # Per-layer precision override (group A)
    if fp16_layers:
        layers = [s.strip() for s in fp16_layers.split(",")]
        for layer in layers:
            lines.append(f"quantization_param({layer}, precision_mode={precision_mode})")

    # Per-layer precision override (group B — e.g. ew_add layers needing different mode)
    if fp16_layers_b:
        layers = [s.strip() for s in fp16_layers_b.split(",")]
        for layer in layers:
            lines.append(f"quantization_param({layer}, precision_mode={precision_mode_b})")

    # Per-layer quantization groups — split weights into N groups for finer-grained scales
    if quantization_groups is not None:
        if quantization_groups_layers:
            layers = [s.strip() for s in quantization_groups_layers.split(",")]
            for layer in layers:
                lines.append(f"quantization_param({layer}, quantization_groups={quantization_groups})")
        else:
            lines.append(f"quantization_param({{conv*}}, quantization_groups={quantization_groups})")

    # Post-quantization: bias correction
    if bias_correction:
        lines.append("post_quantization_optimization(bias_correction, policy=enabled)")

    # Post-quantization: finetuning
    if finetune:
        ft_parts = [
            "post_quantization_optimization(finetune, policy=enabled",
            f"learning_rate={finetune_lr}",
            f"epochs={finetune_epochs}",
            f"dataset_size={finetune_dataset_size}",
        ]
        if finetune_loss_type:
            ft_parts.append(f"def_loss_type={finetune_loss_type}")
        lines.append(", ".join(ft_parts) + ")")

    script = "\n".join(lines)
    return script


def optimize(
    runner: ClientRunner,
    calib_path: str,
    model_script: str,
    har_path: str | None = None,
    normalization_mean: list[float] | None = None,
    normalization_std: list[float] | None = None,
) -> ClientRunner:
    """Quantize model with given model script and calibration data.

    Args:
        runner: Parsed ClientRunner.
        calib_path: Path to .npy file with calibration images (NHWC, float32).
        model_script: Hailo model script string.
        har_path: Path to save the quantized HAR file. If None, not saved.
        normalization_mean: If set, scale calibration data to pre-normalization range.
        normalization_std: If set, scale calibration data to pre-normalization range.

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

    # When normalization is fused into the model, calibration data must be in
    # the pre-normalization range. Undo the normalization: raw = data * std + mean
    if normalization_mean is not None and normalization_std is not None:
        mean = np.array(normalization_mean, dtype=np.float32)
        std = np.array(normalization_std, dtype=np.float32)
        calib_data = calib_data * std + mean
        print(f"Scaled calibration to pre-normalization range:")
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
    parser.add_argument(
        "--calibset-size",
        type=int,
        default=1024,
        help="Number of images for calibration statistics (DFC default=64, we use 1024)",
    )

    # Input normalization (fused into first conv by DFC)
    parser.add_argument(
        "--normalization-mean",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Per-channel mean for normalization fused into graph. E.g. 0 0 0 for [0,255] input.",
    )
    parser.add_argument(
        "--normalization-std",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Per-channel std for normalization fused into graph. E.g. 255 255 255 for [0,255] input.",
    )

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
        help="Comma-separated layer patterns for non-INT8 precision (name kept for compat)",
    )
    parser.add_argument(
        "--precision-mode",
        type=str,
        default="a16_w16",
        choices=["a16_w16", "a16_w8", "a16_w4", "a8_w4"],
        help="Precision mode for --fp16-layers (default: a16_w16)",
    )
    parser.add_argument(
        "--fp16-layers-b",
        type=str,
        default=None,
        help="Second layer group with different precision (e.g. ew_add layers)",
    )
    parser.add_argument(
        "--precision-mode-b",
        type=str,
        default="a16_w16",
        choices=["a16_w16", "a16_w8", "a16_w4", "a8_w4"],
        help="Precision mode for --fp16-layers-b (default: a16_w16)",
    )

    # Post-quantization
    parser.add_argument("--no-bias-correction", action="store_true", help="Disable bias correction")
    parser.add_argument("--finetune", action="store_true", help="Enable post-quantization finetuning")
    parser.add_argument("--finetune-epochs", type=int, default=8, help="Finetuning epochs")
    parser.add_argument("--finetune-lr", type=float, default=0.0001, help="Finetuning learning rate")
    parser.add_argument(
        "--finetune-dataset-size", type=int, default=4096, help="Number of samples for finetuning"
    )
    parser.add_argument(
        "--finetune-loss-type",
        type=str,
        default=None,
        choices=["l2", "l2rel", "l2rel_chw", "ce"],
        help="Loss function for finetuning (l2, l2rel, l2rel_chw, ce). Default: DFC default 'l2rel'.",
    )

    # Quantization groups
    parser.add_argument(
        "--quantization-groups",
        type=int,
        default=None,
        help="Split weight quantization into N groups for finer scales (2-4)",
    )
    parser.add_argument(
        "--quantization-groups-layers",
        type=str,
        default=None,
        help="Layer patterns for quantization_groups (default: all conv layers)",
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
        calibset_size=args.calibset_size,
        normalization_mean=args.normalization_mean,
        normalization_std=args.normalization_std,
        activation_clipping=args.activation_clipping,
        weights_clipping=args.weights_clipping,
        fp16_layers=args.fp16_layers,
        precision_mode=args.precision_mode,
        fp16_layers_b=args.fp16_layers_b,
        precision_mode_b=args.precision_mode_b,
        finetune=args.finetune,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        finetune_dataset_size=args.finetune_dataset_size,
        finetune_loss_type=args.finetune_loss_type,
        bias_correction=not args.no_bias_correction,
        quantization_groups=args.quantization_groups,
        quantization_groups_layers=args.quantization_groups_layers,
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
    runner = optimize(
        runner,
        calib_path=args.calib_data,
        model_script=model_script,
        har_path=quantized_har,
        normalization_mean=args.normalization_mean,
        normalization_std=args.normalization_std,
    )

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
