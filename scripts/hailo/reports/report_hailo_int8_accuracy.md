# Hailo INT8 Quantization Accuracy Report

Investigation of replacing ACT's ResNet18 vision backbone with a Hailo-10H accelerated INT8 version for the Aloha TransferCube task.

## Setup

- **Hardware**: Hailo-10H PCIe, Intel i7-1270P, NVIDIA MX550
- **HailoRT**: 5.2.0, DFC: 5.2.0
- **Task**: `AlohaTransferCube-v0` (bimanual pick-and-transfer)
- **Policy**: ACT with ResNet18 backbone, chunk_size=100, n_action_steps=100
- **Checkpoint**: `lerobot/act_aloha_sim_transfer_cube_human` (migrated)

## Critical Bug Found: ONNX Export Used Wrong Weights

The original `export_resnet18_onnx.py` exported a **fresh ImageNet ResNet18**, not the fine-tuned backbone from the ACT checkpoint. The ACT training fine-tunes the backbone (20/100 parameters differ from ImageNet, `optimizer_lr_backbone=1e-5`).

**Fix**: Updated `export_resnet18_onnx.py` to accept `--policy-path` and load fine-tuned weights from `model.safetensors`.

| Metric | Old (ImageNet weights) | Fixed (fine-tuned weights) |
|--------|----------------------|--------------------------|
| Cosine similarity | 0.707 | 0.997 |
| Mean abs diff | 0.369 | 0.026 |
| Max abs diff | 6.78 | 1.33 |

## Comprehensive Eval Results

### Standard chunk_size=100, n_action_steps=100

| Config | Success Rate | Avg Sum Reward | Time (10 ep) | Notes |
|--------|-------------|---------------|--------------|-------|
| CPU FP32 (baseline) | **70%** (7/10) | 186.4 | 66s | |
| GPU FP32 (NVIDIA MX550) | **70%** (7/10) | 191.4 | 51s | |
| Hailo INT8 opt_level=1 | **0%** (0/50) | 0.0 | 81s | SNR 22.28 dB |
| Hailo INT8 opt_level=2 + BC + QAT | **0%** (0/10) | 0.0 | 112s | SNR 25.33 dB |

### Shorter action chunks (n_action_steps=10)

| Config | Success Rate | Avg Sum Reward | Time (10 ep) | Notes |
|--------|-------------|---------------|--------------|-------|
| CPU FP32 | **20%** (2/10) | 44.8 | 153s | Worse than 100 (trained with 100) |
| Hailo INT8 opt_level=1 | **0%** (0/10) | 0.0 | 151s | |

### Temporal ensembling (n_action_steps=1, coeff=0.01)

| Config | Success Rate | Avg Sum Reward | Time (10 ep) | Notes |
|--------|-------------|---------------|--------------|-------|
| CPU FP32 | **60%** (6/10) | 166.3 | 789s | ~5x slower, but works |
| Hailo INT8 opt_level=1 | **0%** (0/10) | 0.0 | 768s | Even per-step re-querying fails |

## Root Cause: Systematic Action Bias (Not Random Noise)

### Per-Joint Bias Analysis (opt_level=1, 5 seeds)

The INT8 error is **NOT random noise** that averages out -- it introduces **systematic directional biases** on specific joints:

| Joint | Mean Diff | Std | |Mean/Std| | Verdict | Relative Error |
|-------|----------|-----|----------|---------|----------------|
| J4 | +0.006 | 0.002 | 2.5 | BIASED | 1.9% |
| J6 | -0.022 | 0.007 | 3.0 | **BIASED** | **8.8%** |
| J7 | +0.010 | 0.005 | 2.1 | **BIASED** | **20%** |
| J8 | -0.025 | 0.008 | 3.3 | **BIASED** | 1.4% |
| J9 | -0.007 | 0.002 | 3.5 | BIASED | 0.5% |
| Other 9 joints | | | <2 | noisy | |

**Key insight**: Small-magnitude joints (J6, J7) have high relative error (8-20%) because the absolute bias is comparable to the action magnitude. In bimanual manipulation, these joint errors compound through the kinematic chain.

**Why temporal ensembling doesn't help**: It averages random noise but not systematic biases. The quantization pushes certain joints consistently in the wrong direction on every observation.

### Comparison: opt_level=1 vs opt_level=2

| Metric | opt_level=1 | opt_level=2 + BC + QAT |
|--------|-------------|------------------------|
| SNR | 22.28 dB | 25.33 dB |
| Feature cosine | 0.997 | 0.998 |
| Action abs_mean diff | 0.0081 | 0.0088 |
| Action relative error | 1.4% | 1.5% |
| BIASED joints | J4, J6, J7, J8, J9 | J2, J3, J6, J9, J12 |
| Success rate | 0% | 0% |

The bias correction + QAT **changed which joints are biased** but didn't reduce the overall error. J8 improved (biased → noisy) but J3, J12 became newly biased. The ACT transformer decoder amplifies different feature errors into different joint biases depending on the quantization details.

## Error Accumulation Analysis

Ran both policies side-by-side on identical environments (seed=42), logging at each backbone call (every 100 steps):

| Chunk | Step | Obs Image Diff | State Diff | Feature Cosine | Feature MaxDiff | Action MaxDiff |
|-------|------|---------------|------------|---------------|----------------|---------------|
| 0 | 0 | 0.000000 | 0.000000 | 0.997 | 1.27 | 0.031 |
| 1 | 100 | 0.012 | 0.038 | 0.964 | 5.96 | 0.101 |
| 2 | 200 | 0.033 | 0.093 | 0.887 | 9.36 | 0.131 |

**Positive feedback loop**:
```
Small backbone error → biased actions → different robot state →
different camera image → backbone error on different input → larger action error → ...
```

## Performance / Runtime Analysis

**The Hailo-10H provides no acceleration for ACT inference on this task.** In fact, Hailo is slower in the standard configuration.

### Runtime Comparison (10 episodes each)

| Config | Time (10 ep) | Per Episode | vs CPU Baseline |
|--------|-------------|-------------|-----------------|
| **n_action_steps=100** | | | |
| CPU FP32 (baseline) | 66s | 6.6s | — |
| Hailo INT8 opt_level=1 | 81s | 8.1s | **23% slower** |
| Hailo INT8 opt_level=2 | 112s | 11.2s | **70% slower** |
| **Temporal ensembling (n_action_steps=1)** | | | |
| CPU FP32 | 789s | 78.9s | — |
| Hailo INT8 opt_level=1 | 768s | 76.8s | ~3% faster |

### Why No Acceleration?

1. **Backbone is barely called**: With `n_action_steps=100` and 400 steps/episode, the backbone runs only **4 times per episode**. The vast majority of time is spent replaying cached actions and stepping the simulator.
2. **Transformer dominates compute**: The ACT transformer encoder (4 layers) and decoder (1 layer) with dim_model=512 run on CPU regardless. These dominate the per-query compute.
3. **Data transfer overhead**: Each backbone call requires CPU→Hailo→CPU round-trip over PCIe, which adds latency that offsets any compute savings on small models like ResNet18.
4. **No batching**: HailoRT processes one image at a time (batch_size=1). The ResNet18 forward pass on an i7-1270P is already fast enough (~15ms) that the Hailo transfer overhead is not amortized.
5. **opt_level=2 overhead**: The 70% slowdown for opt_level=2 likely reflects additional QAT/bias-correction overhead in the HEF execution or less efficient quantization layout.

### When Hailo Could Help

Temporal ensembling (n_action_steps=1) shows the Hailo is roughly break-even (~3% faster). This mode calls the backbone **every step** (400 calls/episode vs 4), so offloading compute matters more. With a larger backbone (ResNet50, ViT) or higher-resolution inputs, the acceleration benefit would increase. However, given the accuracy failure, this is moot for the current INT8 approach.

## Verification: Non-Backbone Weights

All 133 non-backbone model parameters (encoder, decoder, projection layers, action head) match exactly between the standard and Hailo checkpoints. The issue is purely backbone quantization error.

## ONNX Verification

The ONNX model (FP32) matches PyTorch exactly:
- Cosine similarity: 1.000000
- Max abs diff: 0.000009

This confirms the error is introduced only by INT8 quantization in the HEF, not the ONNX export.

## Quantization Details

### optimization_level=1 (basic)
- **Calibration data**: 1024 frames from Aloha sim, NHWC float32, range [0, 1]
- **SNR**: 22.28 dB
- **HEF size**: 8.5 MB
- **Compile time**: ~5 min

### optimization_level=2 (bias correction + QAT)
- **Bias correction**: 1.5 min (20 conv blocks)
- **QAT distillation**: 40 min (4 epochs x 256 steps, loss: 0.077 → 0.052)
- **SNR**: 25.33 dB (+3 dB improvement)
- **HEF size**: 8.5 MB
- **Total time**: ~47 min

## Compiler Workaround

The Hailo DFC 5.2.0 compiler binary has a `libgflags` ABI mismatch on this system. The system `libgflags` uses `google::` namespace but the compiler needs `gflags::` namespace. **Workaround**:

```bash
BUNDLED_LIBS=".venv/lib/python3.10/site-packages/hailo_tools/or-tools/dependencies/install/lib"
CUDA_VISIBLE_DEVICES="" \
LD_PRELOAD="$BUNDLED_LIBS/libgflags.so.2.2.2:$BUNDLED_LIBS/libglog.so.0.4.0" \
  .venv/bin/python -c "
from hailo_sdk_client import ClientRunner
runner = ClientRunner(har='scripts/resnet18_layer4_quantized.har')
hef = runner.compile()
with open('scripts/resnet18_layer4.hef', 'wb') as f:
    f.write(hef)
"
```

## Conclusion

**INT8 quantization of the ACT ResNet18 backbone is insufficient for the Aloha TransferCube task**, regardless of:
- Quantization optimization level (1 vs 2 with bias correction + QAT)
- Action chunk length (100 vs 10)
- Temporal ensembling (per-step re-query + smoothing)

The fundamental issue is that the ACT transformer decoder is extremely sensitive to small systematic biases in the backbone features. INT8 quantization introduces persistent directional biases (~1.5% relative error overall, but 8-20% on small-magnitude joints) that temporal smoothing cannot correct because they are **not random noise**.

Additionally, **the Hailo-10H provides no latency benefit** for this workload. In the standard configuration (n_action_steps=100), Hailo is actually 23% slower than CPU-only (81s vs 66s per 10 episodes) due to PCIe transfer overhead and the backbone being called only 4 times per episode. Even in temporal ensembling mode (backbone called every step), Hailo is only ~3% faster. ResNet18 at 480x640 is too small a model for the Hailo offloading overhead to pay off, especially when the transformer encoder/decoder still runs on CPU.

## Recommended Next Steps

### 1. Mixed INT8/FP16 on Hailo
Keep sensitive layers (likely layer3/layer4 which produce the final features) in FP16 while quantizing early layers to INT8. The Hailo SDK supports per-layer precision via model script commands. This is the most promising Hailo-based approach.

### 2. ONNX Runtime FP32 on CPU
Use ONNX Runtime instead of PyTorch for FP32 backbone inference on CPU. ONNX Runtime has CPU-specific optimizations (MKL-DNN, XNNPACK) that could be faster than PyTorch without any quantization error. The ONNX export is already verified to be exact.

### 3. Quantization-Aware Training (QAT) in PyTorch
Fine-tune the entire ACT policy (backbone + transformer) with the Hailo INT8 backbone in the training loop. This teaches the transformer decoder to compensate for the quantization bias during training. Most expensive approach but most likely to work.

### 4. INT16 quantization
If the Hailo hardware supports INT16 inference, this would halve the quantization error while still offloading compute.

## File Inventory

All Hailo files are now organized under `scripts/hailo/`.

| File | Status | Purpose |
|------|--------|---------|
| `scripts/hailo/export_resnet18_onnx.py` | **Updated** | Loads fine-tuned weights from checkpoint |
| `scripts/hailo/compile_resnet18_hef.py` | **Updated** | Configurable experiments (opt_level, FP16, clipping, finetune) |
| `scripts/hailo/collect_calibration_images.py` | Unchanged | Collects preprocessed sim frames |
| `scripts/hailo/diagnose_hailo_actions.py` | **Updated** | Per-joint bias analysis (now with CLI args) |
| `scripts/hailo/analyze_layer_noise.py` | **New** | Per-layer SNR noise analysis |
| `scripts/hailo/run_experiments.py` | **New** | Experiment sweep orchestrator |
| `scripts/hailo/artifacts/resnet18_layer4.onnx` | **Rebuilt** | Fine-tuned backbone weights |
| `scripts/hailo/artifacts/resnet18_layer4.hef` | Built | opt_level=1, SNR 22.28 dB |
| `scripts/hailo/artifacts/resnet18_layer4_opt2.hef` | Built | opt_level=2 + BC + QAT, SNR 25.33 dB |
| `scripts/hailo/artifacts/calibration_images.npy` | Unchanged | 1024 frames, 480x640x3, float32 |
| `src/lerobot/policies/act/hailo_backbone.py` | Unchanged | Drop-in HailoBackbone module |
| `src/lerobot/policies/act/configuration_act.py` | Unchanged | use_hailo_backbone / hailo_hef_path fields |
| `src/lerobot/policies/act/modeling_act.py` | Unchanged | Backbone switch logic (line 324) |
| `docs/guide_hailo_acceleration.md` | **Updated** | Paths updated to scripts/hailo/ |

## Environment Notes

- numpy must be 1.26.x (not 2.x) due to scipy 1.12.0 binary compatibility
- Hailo packages installed from `~/Desktop/Hailo_installation/5.2.0/`
- `.venv/` contains all dependencies; use `.venv/bin/python` for all commands
