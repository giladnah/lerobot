# Accelerating ACT with Hailo: Final Report

## Executive Summary

We successfully offloaded the ACT policy's ResNet18 vision backbone to a Hailo-10H AI accelerator while maintaining full task accuracy. The key result:

| Configuration | Success Rate | Avg Reward | Eval Time (10 ep) | HEF Size |
|--------------|-------------|-----------|-------------------|----------|
| CPU FP32 (baseline) | **70%** | 186.4 | 66s | N/A |
| **Hailo uint16 (all layers)** | **70%** | 182.3 | 61s | 30.9 MB |
| Hailo INT8 (all experiments) | 0% | 0.0 | 81-262s | 8.5-30.6 MB |

**Bottom line**: INT8 quantization is fundamentally incompatible with the ACT policy on this task. Running all backbone layers in Hailo's uint16 mode (`a16_w16`) fully restores FP32 accuracy with comparable inference speed.

---

## 1. Background

### 1.1 The Problem

The ACT (Action Chunking with Transformers) policy uses a ResNet18 backbone for visual feature extraction. The backbone processes camera images into feature maps that feed a transformer encoder/decoder to produce robot actions. For real-time robot control, offloading the backbone to a Hailo-10H AI accelerator could reduce CPU load and latency.

### 1.2 Architecture

```
Standard ACT:  Image -> [ResNet18 on CPU/GPU] -> features -> Transformer -> actions
Hailo ACT:     Image -> [ResNet18 on Hailo]   -> features -> Transformer -> actions
```

The `HailoBackbone` module is a drop-in replacement for `IntermediateLayerGetter(ResNet18, {"layer4": "feature_map"})`. It takes `(B, 3, 480, 640)` float32 input and returns `{"feature_map": (B, 512, 15, 20)}` float32 output.

### 1.3 Hardware

- **Hailo device**: Hailo-10H PCIe accelerator
- **CPU**: Intel i7-1270P
- **GPU**: NVIDIA MX550
- **HailoRT/DFC**: v5.2.0
- **Task**: Aloha TransferCube-v0 (bimanual pick-and-transfer in MuJoCo simulation)

---

## 2. What Failed: INT8 Quantization

### 2.1 Results

Every INT8 configuration tested yielded 0% success:

| Config | Success | SNR (dB) | Notes |
|--------|---------|----------|-------|
| INT8 opt_level=1 (basic) | 0% | 22.28 | 50 episodes tested |
| INT8 opt_level=2 + bias correction + QAT | 0% | 25.33 | 4 epochs QAT |
| INT8 + shorter action chunks (n=10) | 0% | — | CPU FP32 gets 20% |
| INT8 + temporal ensembling (n=1, coeff=0.01) | 0% | — | CPU FP32 gets 60% |

### 2.2 Root Cause: Systematic Action Bias

INT8 quantization does not introduce random noise — it introduces **systematic directional biases** on specific joints. The per-joint bias analysis (5 seeds) showed:

| Joint | Mean Diff | |Mean/Std| | Relative Error | Verdict |
|-------|----------|----------|----------------|---------|
| J6 | -0.022 | 3.0 | **8.8%** | BIASED |
| J7 | +0.010 | 2.1 | **20%** | BIASED |
| J8 | -0.025 | 3.3 | 1.4% | BIASED |
| J9 | -0.007 | 3.5 | 0.5% | BIASED |

Small-magnitude joints (J6, J7) are affected most severely because the absolute bias is comparable to the action magnitude. In bimanual manipulation, these compound through the kinematic chain.

**Temporal ensembling cannot fix this** because it averages random noise but not systematic biases. The quantization pushes certain joints consistently in the wrong direction on every observation.

### 2.3 Error Accumulation (Positive Feedback Loop)

Running CPU FP32 and Hailo INT8 side-by-side on identical environments showed rapid divergence:

| Step | Feature Cosine | Action MaxDiff | Observation Diff |
|------|---------------|---------------|-----------------|
| 0 | 0.997 | 0.031 | 0.000 |
| 100 | 0.964 | 0.101 | 0.012 |
| 200 | 0.887 | 0.131 | 0.033 |

```
Small backbone error -> biased actions -> different robot state ->
different camera image -> backbone error on different input -> larger action error -> ...
```

### 2.4 Bias Correction Changes Which Joints Fail, Not Whether They Fail

opt_level=2 with bias correction + QAT **changed which joints** were biased but didn't reduce the overall error:

| | opt_level=1 | opt_level=2 + BC + QAT |
|--|------------|------------------------|
| Biased joints | J4, J6, J7, J8, J9 | J2, J3, J6, J9, J12 |
| Overall action error | 1.4% | 1.5% |
| Success rate | 0% | 0% |

---

## 3. What Worked: uint16 (a16_w16)

### 3.1 Per-Layer Noise Analysis

Before going to full uint16, we tested each ResNet18 layer group individually in uint16 to understand where quantization error originates:

| Config | SNR (dB) | Improvement | Cosine Similarity |
|--------|----------|-------------|-------------------|
| baseline INT8 (all layers) | 18.92 | — | 0.9937 |
| uint16 on layer1 only | 20.04 | +1.13 | 0.9951 |
| uint16 on layer2 only | 19.90 | +0.98 | 0.9950 |
| uint16 on layer3 only | 19.64 | +0.72 | 0.9946 |
| uint16 on layer4 only | 21.09 | +2.17 | 0.9961 |
| **uint16 on ALL layers** | **45.68** | **+26.76** | **0.99999** |

**Key insight**: Quantization error is **uniformly distributed** across all layers. No single layer group provides more than 2.2 dB improvement. The errors interact multiplicatively through the network — all layers must be in uint16 to eliminate the error.

### 3.2 Spatial Error Analysis

Quantization errors are concentrated in the **center** of the feature map (where the robot arm and cube appear), with 13x higher noise power than at the edges. This explains why even small SNR differences have outsized impact on the predicted actions.

### 3.3 Final Result: uint16 on All Layers

| Metric | CPU FP32 | Hailo uint16_all |
|--------|----------|-----------------|
| Success Rate | **70%** (7/10) | **70%** (7/10) |
| Avg Sum Reward | 186.4 | 182.3 |
| Eval Time (10 episodes) | 66s | 61s |
| HEF Size | — | 30.9 MB |
| Hardware Contexts | — | 5 |
| Compilation Time | — | ~13 min |

The uint16_all configuration uses Hailo's `precision_mode=a16_w16` on all conv and elementwise-add layers. This is **16-bit unsigned integer** quantization (NOT IEEE FP16). HailoRT handles the quantization/dequantization internally, presenting float32 I/O to the application.

### 3.4 Compilation Details

The uint16_all HEF required:
- **5 hardware contexts** (vs 1 for INT8) — the model is too large for a single context
- **opt_level=2** with bias correction and 4 epochs QAT (loss: 0.051 -> 0.019)
- **gflags LD_PRELOAD workaround** for DFC 5.2.0 compiler symbol mismatch

---

## 4. Methods and Tools Developed

### 4.1 HailoBackbone Drop-in Module

**File**: `src/lerobot/policies/act/hailo_backbone.py`

A PyTorch `nn.Module` that replaces the standard ResNet18 backbone:
- Initializes Hailo `VDevice` and loads the HEF
- Sets float32 I/O format (HailoRT handles quantization internally)
- Forward: NCHW -> NHWC conversion, per-sample inference, NHWC -> NCHW back
- Returns `{"feature_map": tensor}` matching `IntermediateLayerGetter` output format

Enabled via two config fields in `configuration_act.py`:
```python
use_hailo_backbone: bool = False
hailo_hef_path: str | None = None
```

The model (`modeling_act.py`) switches backbone at init:
```python
if config.use_hailo_backbone:
    from lerobot.policies.act.hailo_backbone import HailoBackbone
    self.backbone = HailoBackbone(config.hailo_hef_path)
```

### 4.2 ONNX Export with Fine-Tuned Weights

**File**: `scripts/hailo/export_resnet18_onnx.py`

Exports ACT's exact ResNet18 backbone to ONNX. Critical fix: must load **fine-tuned weights** from the ACT checkpoint, not fresh ImageNet weights.

```bash
python scripts/hailo/export_resnet18_onnx.py \
    --policy-path outputs/migrated/act_aloha_sim_transfer_cube_human \
    --output scripts/hailo/artifacts/resnet18_layer4.onnx
```

Verified: ONNX matches PyTorch with max abs diff < 1e-5.

### 4.3 Configurable HEF Compilation Pipeline

**File**: `scripts/hailo/compile_resnet18_hef.py`

Full ONNX -> HAR -> HEF pipeline with CLI flags for systematic experiments:

| Flag | Purpose |
|------|---------|
| `--optimization-level` | DFC optimization 0-4 (4=AdaRound) |
| `--fp16-layers` | Layer patterns for `a16_w16` precision |
| `--activation-clipping` | Layer patterns for activation clipping |
| `--weights-clipping` | Layer patterns for weights clipping |
| `--finetune` | Enable post-quantization QAT |
| `--finetune-epochs` | QAT epochs |
| `--experiment-name` | Tag for output artifacts |
| `--dry-run` | Print model script without running |

Example for the winning configuration:
```bash
python scripts/hailo/compile_resnet18_hef.py \
    --optimization-level 2 \
    --fp16-layers "resnet18_layer4/conv1,resnet18_layer4/conv2,...,resnet18_layer4/ew_add8" \
    --finetune --finetune-epochs 4 \
    --experiment-name opt2_uint16_all
```

### 4.4 Per-Layer uint16 Impact Analysis Method

A technique for identifying which layers contribute most to quantization error:

1. Parse ONNX into HAR
2. For each layer group, build a model script with only that group in `a16_w16`
3. Quantize with `runner.optimize()` (opt_level=0 for isolation)
4. Compare FP32 vs quantized inference using `runner.infer_context()`
5. Compute SNR and cosine similarity of output features

**DFC layer names for ResNet18**:

| ResNet Block | DFC Layer Names |
|-------------|----------------|
| Initial conv | conv1 |
| layer1 | conv2-conv5, ew_add1-2 |
| layer2 | conv6-conv10, ew_add3-4 |
| layer3 | conv11-conv15, ew_add5-6 |
| layer4 | conv16-conv20, ew_add7-8 |

**Important**: Normalization layers are fused into convolutions during DFC processing — only use conv and ew_add layer names in `quantization_param()`.

### 4.5 Per-Joint Bias Diagnosis

**File**: `scripts/hailo/diagnose_hailo_actions.py`

Runs the full ACT policy with both CPU FP32 and Hailo backends on identical observations, then computes per-joint mean difference, standard deviation, |mean/std| ratio, and relative error. A joint is flagged as "BIASED" if |mean/std| > 2 across multiple seeds.

### 4.6 Calibration Image Collection

**File**: `scripts/hailo/collect_calibration_images.py`

Collects preprocessed frames from the Aloha simulation environment using the ACT policy's exact preprocessing pipeline (MEAN_STD normalization). Saves 1024 frames as NHWC float32 `.npy` for DFC calibration.

---

## 5. Key Lessons Learned

### 5.1 ACT Is Extremely Sensitive to Feature Quantization

The ACT transformer decoder amplifies small systematic feature biases into large action errors. Even 22-25 dB SNR (seemingly good) causes 0% success because the errors are directional, not random.

### 5.2 Bias Correction Redistributes Error, Doesn't Remove It

DFC's bias correction + QAT at opt_level=2 improved SNR from 22.28 to 25.33 dB but changed which joints were biased rather than fixing the fundamental problem. The total error budget was merely shuffled.

### 5.3 Quantization Error Is Uniformly Distributed

No single ResNet18 layer is a "bottleneck" — error contribution is roughly equal (~1-2 dB per layer group). Selective uint16 on one or two layers cannot solve the problem. This was confirmed by testing all 4 layer groups individually.

### 5.4 Spatial Error Concentration Matters

Quantization errors are 13x higher in the image center (where the robot arm and objects are) than at the edges. This spatial pattern explains why whole-image SNR metrics underestimate the impact on task performance.

### 5.5 The ONNX Export Must Use Fine-Tuned Weights

A critical early bug: the original export used fresh ImageNet ResNet18 weights instead of the fine-tuned backbone from the ACT checkpoint. This caused cosine similarity to drop from 1.0 to 0.707, completely corrupting the features. Always load weights from the trained checkpoint.

### 5.6 DFC `analyze_noise()` Returns None

The DFC built-in `runner.analyze_noise()` runs for ~43 minutes but returns `None` — results are only accessible via the Hailo Model Profiler GUI. Instead, use direct FP32 vs INT8 inference comparison via `runner.infer_context()`.

### 5.7 gflags Workaround for DFC 5.2.0

The Hailo DFC 5.2.0 compiler has a `libgflags` ABI mismatch on some systems. Workaround:

```bash
BUNDLED_LIBS=".venv/lib/python3.10/site-packages/hailo_tools/or-tools/dependencies/install/lib"
LD_PRELOAD="$BUNDLED_LIBS/libgflags.so.2.2.2:$BUNDLED_LIBS/libglog.so.0.4.0" \
    .venv/bin/python scripts/hailo/compile_resnet18_hef.py ...
```

---

## 6. Reproducing the Results

### Step 1: Migrate checkpoint
```bash
python src/lerobot/processor/migrate_policy_normalization.py \
    --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
    --output-dir outputs/migrated/act_aloha_sim_transfer_cube_human
```

### Step 2: Export ONNX
```bash
python scripts/hailo/export_resnet18_onnx.py \
    --policy-path outputs/migrated/act_aloha_sim_transfer_cube_human \
    --output scripts/hailo/artifacts/resnet18_layer4.onnx
```

### Step 3: Collect calibration images
```bash
MUJOCO_GL=egl python scripts/hailo/collect_calibration_images.py \
    --policy-path outputs/migrated/act_aloha_sim_transfer_cube_human \
    --n-frames 1024 --output scripts/hailo/artifacts/calibration_images.npy
```

### Step 4: Compile HEF (uint16 all layers)
```bash
# List all conv and ew_add layers for uint16
UINT16_LAYERS="resnet18_layer4/conv1,resnet18_layer4/conv2,resnet18_layer4/conv3,\
resnet18_layer4/conv4,resnet18_layer4/conv5,resnet18_layer4/conv6,resnet18_layer4/conv7,\
resnet18_layer4/conv8,resnet18_layer4/conv9,resnet18_layer4/conv10,resnet18_layer4/conv11,\
resnet18_layer4/conv12,resnet18_layer4/conv13,resnet18_layer4/conv14,resnet18_layer4/conv15,\
resnet18_layer4/conv16,resnet18_layer4/conv17,resnet18_layer4/conv18,resnet18_layer4/conv19,\
resnet18_layer4/conv20,resnet18_layer4/ew_add1,resnet18_layer4/ew_add2,\
resnet18_layer4/ew_add3,resnet18_layer4/ew_add4,resnet18_layer4/ew_add5,\
resnet18_layer4/ew_add6,resnet18_layer4/ew_add7,resnet18_layer4/ew_add8"

# Compile with gflags workaround
BUNDLED_LIBS=".venv/lib/python3.10/site-packages/hailo_tools/or-tools/dependencies/install/lib"
LD_PRELOAD="$BUNDLED_LIBS/libgflags.so.2.2.2:$BUNDLED_LIBS/libglog.so.0.4.0" \
    .venv/bin/python scripts/hailo/compile_resnet18_hef.py \
    --optimization-level 2 \
    --fp16-layers "$UINT16_LAYERS" \
    --finetune --finetune-epochs 4 \
    --experiment-name opt2_uint16_all
```

### Step 5: Prepare eval checkpoint
```bash
cp -r outputs/migrated/act_aloha_sim_transfer_cube_human \
      outputs/migrated/act_aloha_sim_transfer_cube_human_hailo_uint16_all

# Edit config.json to set:
#   "use_hailo_backbone": true,
#   "hailo_hef_path": "scripts/hailo/artifacts/resnet18_layer4_opt2_uint16_all.hef"
```

### Step 6: Run eval
```bash
MUJOCO_GL=egl .venv/bin/lerobot-eval \
    --policy.type act \
    --policy.pretrained_path outputs/migrated/act_aloha_sim_transfer_cube_human_hailo_uint16_all \
    --env.type aloha --env.task AlohaTransferCube-v0 \
    --eval.n_episodes 10 --eval.batch_size 1
```

Expected: **70% success rate, ~60s eval time.**

---

## 7. File Inventory

### Source (committed)
| File | Purpose |
|------|---------|
| `src/lerobot/policies/act/hailo_backbone.py` | Drop-in HailoBackbone module |
| `src/lerobot/policies/act/configuration_act.py` | `use_hailo_backbone` / `hailo_hef_path` config |
| `src/lerobot/policies/act/modeling_act.py` | Backbone switch logic (line ~324) |

### Scripts
| File | Purpose |
|------|---------|
| `scripts/hailo/export_resnet18_onnx.py` | ONNX export with fine-tuned weights |
| `scripts/hailo/compile_resnet18_hef.py` | Configurable ONNX -> HEF pipeline |
| `scripts/hailo/collect_calibration_images.py` | Calibration data from sim |
| `scripts/hailo/diagnose_hailo_actions.py` | Per-joint bias analysis |
| `scripts/hailo/analyze_layer_noise.py` | Per-layer SNR analysis |

### Artifacts
| File | Size | Description |
|------|------|-------------|
| `scripts/hailo/artifacts/resnet18_layer4.onnx` | 44 MB | Fine-tuned backbone ONNX |
| `scripts/hailo/artifacts/resnet18_layer4_opt2_uint16_all.hef` | 30.9 MB | Winning HEF (uint16 all layers) |
| `scripts/hailo/artifacts/calibration_images.npy` | 3.7 GB | 1024 calibration frames |

### Reports
| File | Description |
|------|-------------|
| `scripts/hailo/reports/report_hailo_acceleration_final.md` | This report |
| `scripts/hailo/reports/report_experiments.md` | Experiment-by-experiment results |
| `scripts/hailo/reports/report_hailo_int8_accuracy.md` | Detailed INT8 failure analysis |

---

## 8. Trade-offs and Future Work

### 8.1 Current Trade-offs
- **HEF size**: 30.9 MB vs 8.5 MB for INT8 (3.6x larger)
- **Hardware contexts**: 5 vs 1 (uses more on-chip resources)
- **Inference speed**: Comparable to CPU FP32 (61s vs 66s for 10 episodes)
- **Memory**: Hailo device memory usage is higher with uint16

### 8.2 Potential Improvements
- **Mixed INT8/uint16**: Keep early layers (layer1/2) in INT8 where errors are smaller, use uint16 only for layer3/4. Per-layer analysis suggests this would recover ~75% of the SNR improvement while using fewer resources. However, layer4-only uint16 still yielded 0% success, so this may require layer3+layer4 at minimum.
- **ONNX Runtime FP32**: For deployments without Hailo hardware, ONNX Runtime with CPU optimizations (MKL-DNN) could be faster than PyTorch FP32.
- **Quantization-Aware Training**: Fine-tune the entire ACT policy (backbone + transformer) with quantized backbone in the training loop, teaching the decoder to compensate for quantization bias.
- **Hailo compiler_optimization_level=max**: The compiler suggested this flag could improve inference performance at the cost of compilation time.
