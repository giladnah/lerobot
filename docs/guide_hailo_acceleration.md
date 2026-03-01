# Accelerating ACT with Hailo

Guide for offloading the ACT policy's ResNet18 vision backbone to a Hailo AI accelerator. This replaces the PyTorch ResNet18 inference with a pre-compiled HEF running on dedicated hardware, freeing the CPU/GPU for the rest of the pipeline.

## Goal

Replace the ResNet18 backbone in ACT's vision pipeline with Hailo-accelerated inference to reduce latency for real-time robot control. The backbone is the most compute-intensive part of the ACT inference path.

## Architecture

```
Standard ACT:     Image -> [ResNet18 on CPU/GPU] -> features -> Transformer Encoder/Decoder -> actions
Hailo ACT:        Image -> [ResNet18 on Hailo]   -> features -> Transformer Encoder/Decoder -> actions
```

The `HailoBackbone` is a drop-in replacement for `IntermediateLayerGetter(ResNet18, {"layer4": "feature_map"})`. It takes `(B, 3, H, W)` float32 input and returns `{"feature_map": (B, 512, H/32, W/32)}` float32 output -- identical interface.

## Hardware Setup

- **Hailo device**: Hailo-10H PCIe accelerator (device `0000:3d:00.0`)
- **HailoRT version**: 5.2.0
- **Driver**: `hailort-pcie-driver` 5.2.0

Verify the device is detected:

```bash
hailortcli scan
# Expected: Hailo Devices: [-] Device: 0000:3d:00.0
```

## File Inventory

### Source files (in repo)

| File | Purpose |
|------|---------|
| `src/lerobot/policies/act/hailo_backbone.py` | `HailoBackbone` nn.Module -- drop-in replacement for ResNet18 backbone |
| `src/lerobot/policies/act/configuration_act.py` | ACT config with `use_hailo_backbone` and `hailo_hef_path` fields |
| `src/lerobot/policies/act/modeling_act.py` | Model init switches backbone based on config (line 324) |

### Scripts (in `scripts/hailo/`)

| File | Purpose |
|------|---------|
| `scripts/hailo/export_resnet18_onnx.py` | Export ACT's exact ResNet18 (with FrozenBatchNorm2d, layer4) to ONNX |
| `scripts/hailo/compile_resnet18_hef.py` | Compile ONNX to Hailo HEF via `hailo_sdk_client` (configurable optimization experiments) |
| `scripts/hailo/collect_calibration_images.py` | Collect preprocessed frames from Aloha sim for quantization calibration |
| `scripts/hailo/analyze_layer_noise.py` | Per-layer SNR noise analysis for identifying sensitive layers |
| `scripts/hailo/diagnose_hailo_actions.py` | Per-joint action bias analysis (CPU FP32 vs Hailo INT8) |
| `scripts/hailo/run_experiments.py` | Orchestrates systematic quantization experiment sweep |

### Pre-built artifacts (in `scripts/hailo/artifacts/`)

| File | Size | Description |
|------|------|-------------|
| `scripts/hailo/artifacts/resnet18_layer4.onnx` | ~44MB | ONNX model (input: 1x3x480x640, output: 1x512x15x20) |
| `scripts/hailo/artifacts/resnet18_layer4.hef` | ~8.5MB | Compiled Hailo HEF for Hailo-10H |
| `scripts/hailo/artifacts/calibration_images.npy` | ~3.7GB | 1024 preprocessed frames (480x640x3 float32, NHWC, range [0, 1]) |

### Reports (in `scripts/hailo/reports/`)

| File | Description |
|------|-------------|
| `scripts/hailo/reports/report_hailo_int8_accuracy.md` | Comprehensive INT8 quantization findings |
| `scripts/hailo/reports/report_experiments.md` | Systematic optimization experiment results |
| `scripts/hailo/reports/report_layer_noise_analysis.md` | Per-layer SNR analysis results |

### Tests (in `scripts/hailo/tests/`)

| File | Description |
|------|-------------|
| `scripts/hailo/tests/test_onnx_export.py` | Verify ONNX matches PyTorch output |
| `scripts/hailo/tests/test_hef_accuracy.py` | Verify HEF vs ONNX feature similarity |
| `scripts/hailo/tests/test_backbone_integration.py` | Verify HailoBackbone drop-in works |

### Migrated checkpoints (in `outputs/migrated/`)

| Path | Task |
|------|------|
| `outputs/migrated/act_aloha_sim_transfer_cube_human/` | TransferCube (pick & transfer) |
| `outputs/migrated/act_aloha_sim_insertion_human/` | Insertion (peg insertion) |

These were migrated from HF Hub checkpoints using `src/lerobot/processor/migrate_policy_normalization.py`. They contain:
- `config.json` -- ACT config (resnet18, dim_model=512, chunk_size=100, 1 camera at 480x640)
- `model.safetensors` -- model weights (~197MB)
- `policy_preprocessor.json` + `.safetensors` -- MEAN_STD normalization stats
- `policy_postprocessor.json` + `.safetensors` -- action denormalization

## Running Evaluation

### Standard CPU/GPU baseline (no Hailo)

```bash
# TransferCube on CPU
MUJOCO_GL=egl lerobot-eval \
  --policy.path=outputs/migrated/act_aloha_sim_transfer_cube_human \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.device=cpu

# TransferCube on GPU
MUJOCO_GL=egl lerobot-eval \
  --policy.path=outputs/migrated/act_aloha_sim_transfer_cube_human \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.device=cuda
```

### With Hailo backbone

The Hailo backbone is enabled via two config fields. These can be set by modifying the checkpoint's `config.json` or (once CLI override is supported) via command line:

1. Edit the checkpoint config to enable Hailo:

```bash
# Make a copy for Hailo evaluation
cp -r outputs/migrated/act_aloha_sim_transfer_cube_human outputs/migrated/act_aloha_sim_transfer_cube_human_hailo

# Edit config.json -- add/set these fields:
#   "use_hailo_backbone": true,
#   "hailo_hef_path": "scripts/hailo/artifacts/resnet18_layer4.hef"
```

2. Run eval with the Hailo-enabled checkpoint:

```bash
MUJOCO_GL=egl lerobot-eval \
  --policy.path=outputs/migrated/act_aloha_sim_transfer_cube_human_hailo \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.device=cpu
```

Note: With Hailo, `--policy.device=cpu` is typical since the backbone is offloaded and the remaining transformer is lightweight enough for CPU.

## Baseline Results (GPU -- NVIDIA MX550)

From `docs/guide_act_aloha_eval.md`:

| Task | Success Rate | Avg Sum Reward | Avg Max Reward | Time |
|------|-------------|---------------|----------------|------|
| TransferCube | 70% (7/10) | 191.4 | 3.1 | 50.8s |
| Insertion | 20% (2/10) | 268.3 | 2.4 | 61.2s |

## Reproducing the Pipeline from Scratch

If you need to rebuild the HEF (e.g., for different input resolution or Hailo hardware):

### Step 1: Migrate checkpoint (if not already done)

```bash
python src/lerobot/processor/migrate_policy_normalization.py \
  --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
  --output-dir outputs/migrated/act_aloha_sim_transfer_cube_human
```

### Step 2: Export ResNet18 to ONNX

```bash
python scripts/hailo/export_resnet18_onnx.py \
  --height 480 --width 640 \
  --output scripts/hailo/artifacts/resnet18_layer4.onnx
```

Output: `resnet18_layer4.onnx` (input `[1, 3, 480, 640]`, output `[1, 512, 15, 20]`).

### Step 3: Collect calibration images

```bash
MUJOCO_GL=egl python scripts/hailo/collect_calibration_images.py \
  --policy-path outputs/migrated/act_aloha_sim_transfer_cube_human \
  --n-frames 1024 \
  --output scripts/hailo/artifacts/calibration_images.npy
```

Captures 1024 preprocessed frames from the Aloha sim (NHWC float32).

### Step 4: Compile HEF

```bash
python scripts/hailo/compile_resnet18_hef.py \
  --experiment-name baseline
```

For advanced experiments with FP16 layers, clipping, or finetuning:

```bash
python scripts/hailo/compile_resnet18_hef.py \
  --optimization-level 4 \
  --fp16-layers "resnet18_layer4/layer4*" \
  --finetune \
  --experiment-name fp16_layer4_finetune
```

See `python scripts/hailo/compile_resnet18_hef.py --help` for all options.

Requires `hailo_sdk_client` (from `hailo_dataflow_compiler` wheel). Runs parse -> INT8 quantize -> compile.

### Step 5 (optional): Run experiment sweep

```bash
python scripts/hailo/run_experiments.py --skip-eval
```

This runs all 6 predefined experiments and generates a report at `scripts/hailo/reports/report_experiments.md`.

## Implementation Details

### Config fields (`configuration_act.py`)

```python
# Hailo accelerator.
use_hailo_backbone: bool = False
hailo_hef_path: str | None = None
```

Validation in `__post_init__`: requires `hailo_hef_path` when `use_hailo_backbone=True`, skips ResNet name validation when using Hailo.

### Model integration (`modeling_act.py`, lines 323-341)

```python
if config.use_hailo_backbone:
    from lerobot.policies.act.hailo_backbone import HailoBackbone
    self.backbone = HailoBackbone(config.hailo_hef_path)
    backbone_feature_dim = 512
else:
    # Standard torchvision ResNet18
    ...
```

The lazy import means `hailo_platform` is only required when Hailo is actually enabled.

### HailoBackbone (`hailo_backbone.py`)

- Initializes a `VDevice` and loads the HEF
- Sets float32 I/O format (HailoRT handles INT8 quant/dequant internally)
- Forward: converts NCHW -> NHWC, runs inference per sample, converts back
- Batch processing is sequential (Hailo processes one sample at a time)
- Returns `{"feature_map": tensor}` matching `IntermediateLayerGetter` output format

### System dependencies

| Package | Source | Purpose |
|---------|--------|---------|
| `hailort` (system) | `dpkg` / Hailo SDK | PCIe driver + runtime (v5.2.0) |
| `hailo_platform` (Python) | HailoRT wheel | Python bindings for inference |
| `hailo_sdk_client` (Python) | Hailo Dataflow Compiler | HEF compilation only (not needed at runtime) |
| `gym-aloha` (Python) | `uv sync --extra aloha` | Aloha MuJoCo simulation |

### Installing Hailo SDK

Download Hailo packages using the official artifact downloader:

```bash
# Download the downloader script
curl -O https://dev-public.hailo.ai/scripts/common/artifacts_downloader.sh
chmod +x artifacts_downloader.sh

# Download HailoRT for Hailo-10H (H10), version 5.2.0
./artifacts_downloader.sh -d H10 -p hailort -v 5.2.0

# Install the Python wheel into the venv
pip install <downloaded_path>/hailort-5.2.0-cp310-cp310-linux_x86_64.whl

# For HEF compilation, also install the Dataflow Compiler
pip install <downloaded_path>/hailo_dataflow_compiler-*.whl
```

The downloader supports `-a arm64` for ARM platforms. Run `./artifacts_downloader.sh -h` for all options.

## Known Limitations

- Batch inference is sequential -- the Hailo device processes one image at a time
- HEF is compiled for fixed input size (480x640) -- different resolutions need recompilation
- HEF is hardware-specific (compiled for `hailo10h`) -- different Hailo chips need recompilation
- INT8 quantization may introduce small accuracy differences vs FP32 PyTorch backbone

## Current Status

**SOLVED**: uint16 (a16_w16) on ALL backbone layers achieves 70% success on TransferCube -- matching the FP32 baseline. INT8 quantization is fundamentally insufficient for ACT (0% success at all optimization levels). See the [Final Report](scripts/hailo/reports/report_hailo_acceleration_final.md) for complete findings.

| Configuration | Success Rate | Eval Time | HEF Size |
|--------------|-------------|-----------|----------|
| CPU FP32 (baseline) | 70% | 66s | N/A |
| Hailo uint16 (all layers) | 70% | 61s | 30.9 MB |
| Hailo INT8 (any config) | 0% | 81-262s | 8.5-30.6 MB |

## TODO

- [ ] Benchmark Hailo vs CPU vs GPU backbone latency (isolated, not end-to-end)
- [ ] Support CLI override for `use_hailo_backbone` / `hailo_hef_path` (avoid config.json editing)
- [ ] Explore ONNX Runtime FP32 as alternative acceleration path
- [ ] Test mixed INT8/uint16 (layers 1-2 INT8, layers 3-4 uint16) to reduce HEF size
- [ ] Test on real robot hardware (not just simulation)
