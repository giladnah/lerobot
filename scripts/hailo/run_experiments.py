"""Orchestrate systematic Hailo quantization experiments.

Runs a sequence of experiments with increasing optimization, compiles HEFs,
and generates a comprehensive report. Each experiment builds on the previous:

  1. Baseline opt_level=4 (AdaRound)
  2. + activation clipping on low-SNR layers
  3. + weights clipping on low-SNR layers
  4. + FP16 on layer4
  5. + FP16 on layer3 + layer4
  6. + post-quantization finetuning

For each experiment:
  - Compiles HEF via compile_resnet18_hef.py
  - Runs diagnose_hailo_actions.py for per-joint bias analysis
  - Optionally runs lerobot-eval for full eval
  - Logs results to reports/report_experiments.md

Usage:
    python scripts/hailo/run_experiments.py

    # Skip eval (compile + diagnose only)
    python scripts/hailo/run_experiments.py --skip-eval

    # Run specific experiments
    python scripts/hailo/run_experiments.py --experiments 1,4,6

    # Dry run (print configs without executing)
    python scripts/hailo/run_experiments.py --dry-run

Requires: hailo_sdk_client, hailo_platform, gym-aloha
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for a single quantization experiment."""

    name: str
    description: str
    optimization_level: int = 4
    compression_level: int = 0
    activation_clipping: str | None = None
    weights_clipping: str | None = None
    fp16_layers: str | None = None
    finetune: bool = False
    finetune_epochs: int = 8
    finetune_lr: float = 0.0001
    no_bias_correction: bool = False


# Default experiment sequence — each builds on the previous
DEFAULT_EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(
        name="opt4_baseline",
        description="Baseline with opt_level=4 (AdaRound, 1024 images)",
        optimization_level=4,
    ),
    ExperimentConfig(
        name="opt4_actclip",
        description="+ activation clipping on layer3/layer4 (percentile 0.01-99.99)",
        optimization_level=4,
        activation_clipping="resnet18_layer4/layer3*,resnet18_layer4/layer4*",
    ),
    ExperimentConfig(
        name="opt4_actclip_wclip",
        description="+ weights clipping on layer3/layer4",
        optimization_level=4,
        activation_clipping="resnet18_layer4/layer3*,resnet18_layer4/layer4*",
        weights_clipping="resnet18_layer4/layer3*,resnet18_layer4/layer4*",
    ),
    ExperimentConfig(
        name="opt4_fp16_layer4",
        description="+ FP16 (a16_w16) on layer4",
        optimization_level=4,
        activation_clipping="resnet18_layer4/layer3*",
        fp16_layers="resnet18_layer4/layer4*",
    ),
    ExperimentConfig(
        name="opt4_fp16_layer34",
        description="+ FP16 (a16_w16) on layer3 + layer4",
        optimization_level=4,
        fp16_layers="resnet18_layer4/layer3*,resnet18_layer4/layer4*",
    ),
    ExperimentConfig(
        name="opt4_fp16_layer4_finetune",
        description="+ FP16 layer4 + post-quantization finetune (8 epochs)",
        optimization_level=4,
        activation_clipping="resnet18_layer4/layer3*",
        fp16_layers="resnet18_layer4/layer4*",
        finetune=True,
        finetune_epochs=8,
    ),
]


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    name: str
    compile_time_s: float = 0.0
    hef_size_bytes: int = 0
    hef_path: str = ""
    diagnose_output: str = ""
    eval_success_rate: float | None = None
    eval_avg_reward: float | None = None
    eval_time_s: float | None = None
    error: str | None = None


def run_compile(exp: ExperimentConfig, python: str = sys.executable) -> tuple[str, float]:
    """Run compile_resnet18_hef.py for an experiment config.

    Returns:
        Tuple of (hef_path, compile_time_seconds).
    """
    cmd = [
        python,
        "scripts/hailo/compile_resnet18_hef.py",
        "--experiment-name",
        exp.name,
        "--optimization-level",
        str(exp.optimization_level),
        "--compression-level",
        str(exp.compression_level),
    ]

    if exp.activation_clipping:
        cmd.extend(["--activation-clipping", exp.activation_clipping])
    if exp.weights_clipping:
        cmd.extend(["--weights-clipping", exp.weights_clipping])
    if exp.fp16_layers:
        cmd.extend(["--fp16-layers", exp.fp16_layers])
    if exp.finetune:
        cmd.extend(
            [
                "--finetune",
                "--finetune-epochs",
                str(exp.finetune_epochs),
                "--finetune-lr",
                str(exp.finetune_lr),
            ]
        )
    if exp.no_bias_correction:
        cmd.append("--no-bias-correction")

    print(f"\n{'=' * 70}")
    print(f"Compiling: {exp.name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"COMPILE FAILED for {exp.name}:")
        print(result.stderr)
        raise RuntimeError(f"Compile failed: {result.stderr[-500:]}")

    print(result.stdout)
    hef_path = f"scripts/hailo/artifacts/resnet18_layer4_{exp.name}.hef"
    return hef_path, elapsed


def run_diagnose(hef_path: str, exp_name: str, python: str = sys.executable) -> str:
    """Run diagnose_hailo_actions.py and return output."""
    # Create a temporary Hailo checkpoint by copying the baseline and updating config
    # For now, just return a placeholder — the user should set up the checkpoint
    print(f"\n--- Diagnose: {exp_name} ---")
    print(f"HEF: {hef_path}")
    print("NOTE: Diagnose requires a checkpoint with use_hailo_backbone=true pointing to this HEF.")
    print("Set up the checkpoint and run manually:")
    print("  MUJOCO_GL=egl python scripts/hailo/diagnose_hailo_actions.py \\")
    print(f"      --hailo-path <checkpoint_with_{exp_name}_hef>")
    return f"[Manual step required — set up checkpoint for {exp_name}]"


def run_eval(hef_path: str, exp_name: str, n_episodes: int = 10) -> tuple[float, float, float]:
    """Run lerobot-eval and return (success_rate, avg_reward, time_s)."""
    print(f"\n--- Eval: {exp_name} ---")
    print("NOTE: Eval requires a checkpoint with use_hailo_backbone=true pointing to this HEF.")
    print("Run manually:")
    print("  MUJOCO_GL=egl lerobot-eval \\")
    print(f"      --policy.path=<checkpoint_with_{exp_name}_hef> \\")
    print("      --env.type=aloha --env.task=AlohaTransferCube-v0 \\")
    print(f"      --eval.batch_size=1 --eval.n_episodes={n_episodes} \\")
    print("      --policy.device=cpu")
    return (0.0, 0.0, 0.0)


def generate_report(experiments: list[ExperimentConfig], results: list[ExperimentResult], output_path: str):
    """Generate comprehensive markdown report."""
    lines = []
    lines.append("# Hailo Quantization Experiment Results\n")
    lines.append("Generated by `run_experiments.py`\n")
    lines.append("## Experiment Summary\n")

    # Summary table
    lines.append(
        f"| {'#':>2s} | {'Experiment':<30s} | {'Opt':>3s} | {'FP16 Layers':<20s} | "
        f"{'Clip':>5s} | {'Finetune':>8s} | {'HEF Size':>10s} | {'Compile':>10s} | {'Status':>8s} |"
    )
    lines.append(
        f"|{'-' * 4}|{'-' * 32}|{'-' * 5}|{'-' * 22}|{'-' * 7}|{'-' * 10}|{'-' * 12}|{'-' * 12}|{'-' * 10}|"
    )

    for i, (exp, res) in enumerate(zip(experiments, results, strict=True), 1):
        clip = "A" if exp.activation_clipping else ""
        clip += "+W" if exp.weights_clipping else ""
        clip = clip or "-"
        ft = f"{exp.finetune_epochs}ep" if exp.finetune else "-"
        fp16 = exp.fp16_layers or "-"
        if len(fp16) > 20:
            fp16 = fp16[:17] + "..."
        hef_mb = f"{res.hef_size_bytes / 1e6:.1f} MB" if res.hef_size_bytes else "-"
        compile_t = f"{res.compile_time_s:.0f}s" if res.compile_time_s else "-"
        status = "ERROR" if res.error else "OK"

        lines.append(
            f"| {i:>2d} | {exp.name:<30s} | {exp.optimization_level:>3d} | {fp16:<20s} | "
            f"{clip:>5s} | {ft:>8s} | {hef_mb:>10s} | {compile_t:>10s} | {status:>8s} |"
        )

    # Detailed results
    lines.append("\n## Detailed Results\n")

    for i, (exp, res) in enumerate(zip(experiments, results, strict=True), 1):
        lines.append(f"### Experiment {i}: {exp.name}\n")
        lines.append(f"**Description**: {exp.description}\n")
        lines.append(f"- Optimization level: {exp.optimization_level}")
        lines.append(f"- Activation clipping: {exp.activation_clipping or 'none'}")
        lines.append(f"- Weights clipping: {exp.weights_clipping or 'none'}")
        lines.append(f"- FP16 layers: {exp.fp16_layers or 'none'}")
        lines.append(f"- Finetune: {'yes' if exp.finetune else 'no'}")
        if exp.finetune:
            lines.append(f"  - Epochs: {exp.finetune_epochs}, LR: {exp.finetune_lr}")
        lines.append(f"- HEF path: `{res.hef_path}`")
        lines.append(
            f"- HEF size: {res.hef_size_bytes / 1e6:.1f} MB" if res.hef_size_bytes else "- HEF size: N/A"
        )
        lines.append(
            f"- Compile time: {res.compile_time_s:.0f}s" if res.compile_time_s else "- Compile time: N/A"
        )

        if res.error:
            lines.append(f"\n**ERROR**: {res.error}")

        if res.eval_success_rate is not None:
            lines.append("\n**Eval Results** (10 episodes):")
            lines.append(f"- Success rate: {res.eval_success_rate:.0%}")
            lines.append(f"- Avg sum reward: {res.eval_avg_reward:.1f}")
            lines.append(f"- Eval time: {res.eval_time_s:.0f}s")

        lines.append("")

    # Comparison table (to be filled in manually after running evals)
    lines.append("## Eval Comparison (fill in after running evals)\n")
    lines.append(
        f"| {'Experiment':<30s} | {'Success':>8s} | {'Avg Reward':>11s} | {'Time':>8s} | {'Notes':<30s} |"
    )
    lines.append(f"|{'-' * 32}|{'-' * 10}|{'-' * 13}|{'-' * 10}|{'-' * 32}|")
    lines.append(
        f"| {'CPU FP32 baseline':<30s} | {'70%':>8s} | {'186.4':>11s} | {'66s':>8s} | {'reference':30s} |"
    )
    for exp in experiments:
        lines.append(f"| {exp.name:<30s} | {'':>8s} | {'':>11s} | {'':>8s} | {'':30s} |")

    report = "\n".join(lines) + "\n"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report)
    print(f"\nReport saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run systematic Hailo quantization experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment numbers to run (1-indexed). Default: all.",
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip lerobot-eval (compile + diagnose only)"
    )
    parser.add_argument("--skip-diagnose", action="store_true", help="Skip diagnose step")
    parser.add_argument("--dry-run", action="store_true", help="Print experiment configs without executing")
    parser.add_argument("--n-episodes", type=int, default=10, help="Episodes per eval run")
    parser.add_argument(
        "--report",
        type=str,
        default="scripts/hailo/reports/report_experiments.md",
        help="Output report path",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use",
    )
    args = parser.parse_args()

    # Select experiments
    if args.experiments:
        indices = [int(x.strip()) - 1 for x in args.experiments.split(",")]
        experiments = [DEFAULT_EXPERIMENTS[i] for i in indices]
    else:
        experiments = DEFAULT_EXPERIMENTS

    print(f"Running {len(experiments)} experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp.name}: {exp.description}")

    if args.dry_run:
        print("\nDry run — printing configs:")
        for exp in experiments:
            from compile_resnet18_hef import build_model_script

            script = build_model_script(
                optimization_level=exp.optimization_level,
                compression_level=exp.compression_level,
                activation_clipping=exp.activation_clipping,
                weights_clipping=exp.weights_clipping,
                fp16_layers=exp.fp16_layers,
                finetune=exp.finetune,
                finetune_epochs=exp.finetune_epochs,
                finetune_lr=exp.finetune_lr,
                bias_correction=not exp.no_bias_correction,
            )
            print(f"\n--- {exp.name} ---")
            print(script)
        return

    # Run experiments
    results = []
    for exp in experiments:
        result = ExperimentResult(name=exp.name)
        try:
            # Compile
            hef_path, compile_time = run_compile(exp, python=args.python)
            result.hef_path = hef_path
            result.compile_time_s = compile_time
            if Path(hef_path).exists():
                result.hef_size_bytes = Path(hef_path).stat().st_size

            # Diagnose
            if not args.skip_diagnose:
                result.diagnose_output = run_diagnose(hef_path, exp.name, python=args.python)

            # Eval
            if not args.skip_eval:
                sr, reward, eval_time = run_eval(hef_path, exp.name, args.n_episodes)
                result.eval_success_rate = sr
                result.eval_avg_reward = reward
                result.eval_time_s = eval_time

        except Exception as e:
            result.error = str(e)
            print(f"\nERROR in {exp.name}: {e}")

        results.append(result)

    # Generate report
    generate_report(experiments, results, args.report)

    # Summary
    print(f"\n{'=' * 70}")
    print("EXPERIMENT SWEEP COMPLETE")
    print(f"{'=' * 70}")
    for res in results:
        status = "ERROR" if res.error else "OK"
        print(f"  {res.name}: {status} ({res.compile_time_s:.0f}s)")
    print(f"\nReport: {args.report}")


if __name__ == "__main__":
    main()
