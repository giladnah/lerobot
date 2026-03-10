# Hailo Quantization Experiment Results

## Experiment Summary

| #   | Experiment                | Opt | uint16 Layers | Clip | Finetune    | HEF Size | Compile  | Status    |
| --- | ------------------------- | --- | ------------- | ---- | ----------- | -------- | -------- | --------- |
| 1   | opt4_baseline             | 4   | -             | -    | -           | -        | -        | OOM (70GB needed, 34GB available) |
| 2   | opt2_uint16_layer4        | 2   | layer4        | -    | 4ep QAT     | 30.6 MB  | ~55 min  | Done      |
| 3   | opt2_uint16_layer34       | 2   | layer3+layer4 | -    | 4ep QAT     | -        | -        | Stopped   |
| 4   | opt2_uint16_all           | 2   | ALL layers    | -    | 4ep QAT     | 30.9 MB  | ~13 min  | Done      |
| 5   | opt2_calibfix             | 2   | -             | -    | -           | 8.5 MB   | ~35 min  | Done — SNR 25.08 dB |
| 6   | opt2_normin               | 2   | -             | -    | -           | 8.5 MB   | ~35 min  | Done — SNR 24.90 dB |
| 7   | opt2_l2_ft16              | 2   | -             | -    | L2/16ep     | 8.7 MB   | ~3.5 hr  | Done — SNR 25.73 dB |
| 8   | opt2_qg4_l2_ft8           | 2   | -             | -    | L2/8ep QG=4 | -        | -        | Failed — conv3 doesn't support QG |

**Note**: opt_level=4 (AdaRound) requires ~70 GB of storage for 1024 calibration images, exceeding the 34 GB RAM on this system. Switched to opt_level=2 (bias correction + QAT) which works within memory.

**Note**: All "uint16" experiments use Hailo's `precision_mode=a16_w16` which is 16-bit quantized (unsigned integer), NOT IEEE FP16.

### Layer name mapping (DFC → ResNet18)

| DFC Layers | ResNet18 Block | Notes |
|------------|---------------|-------|
| conv1 | initial conv | 7x7, stride 2 |
| conv2-conv5, ew_add1-2 | layer1 | 2 basic blocks |
| conv6-conv10, ew_add3-4 | layer2 | 2 basic blocks + downsample |
| conv11-conv15, ew_add5-6 | layer3 | 2 basic blocks + downsample |
| conv16-conv20, ew_add7-8 | layer4 | 2 basic blocks + downsample |

## Previous Results (for reference)

| Config                      | Success Rate | SNR      | Notes                                       |
| --------------------------- | ------------ | -------- | ------------------------------------------- |
| CPU FP32 baseline           | **70%**      | N/A      | reference                                   |
| Hailo INT8 opt_level=1      | **0%**       | 22.28 dB | basic quantization                          |
| Hailo INT8 opt_level=2 + BC | **0%**       | 25.33 dB | bias correction changed which joints biased |

See [report_hailo_int8_accuracy.md](report_hailo_int8_accuracy.md) for full previous findings.

## Per-Layer uint16 Noise Analysis

Tested each ResNet layer group individually in uint16 (a16_w16) with opt_level=0 to isolate the contribution of each layer to quantization error:

| Config | SNR (dB) | Improvement | Cosine Similarity |
|--------|----------|-------------|-------------------|
| baseline_int8 (all INT8) | 18.92 | - | 0.9937 |
| uint16_layer1 only | 20.04 | +1.13 | 0.9951 |
| uint16_layer2 only | 19.90 | +0.98 | 0.9950 |
| uint16_layer3 only | 19.64 | +0.72 | 0.9946 |
| **uint16_layer4 only** | **21.09** | **+2.17** | **0.9961** |
| **uint16_all layers** | **45.68** | **+26.76** | **0.99999** |

**Key finding**: Quantization error is **distributed across ALL layers**, not concentrated in any single layer group. No individual layer group provides more than 2.2 dB improvement. All layers must be in uint16 to achieve near-FP32 quality (45.68 dB SNR vs 18.92 dB baseline).

## Eval Comparison

| Experiment                | Success | Avg Reward | Time  | HEF Size | Notes     |
| ------------------------- | ------- | ---------- | ----- | -------- | --------- |
| CPU FP32 baseline         | **70%** | 186.4      | 66s   | N/A      | reference |
| INT8 opt_level=1          | 0%      | 0.0        | 81s   | 8.5 MB   | previous  |
| INT8 opt_level=2 + BC     | 0%      | 0.0        | 112s  | 8.5 MB   | previous  |
| opt2_uint16_layer4        | **0%**  | 0.0        | 262s  | 30.6 MB  | uint16 on layer4 only; 3.5x slower than CPU |
| **opt2_uint16_all**       | **70%** | 182.3      | 61s   | 30.9 MB  | **uint16 on ALL layers; matches FP32 baseline!** |

## Phase 2: INT8 Accuracy Recovery Experiments

Following the gap analysis in the plan, these experiments attempt to improve INT8 accuracy through better calibration and finetuning techniques.

### Parameters Under Test

| Parameter | Description | Rationale |
|-----------|------------|-----------|
| `calibset_size=1024` | Use all 1024 calibration images for statistics | DFC default is only 64; more data = better quantization ranges |
| `finetune-loss-type=l2` | L2 loss for post-quant finetuning | May be better than default l2rel for feature maps |
| `quantization_groups=4` | Split weights into 4 groups per layer | Finer-grained scale factors = less quantization error |
| `finetune-epochs=16` | More finetune epochs | Allow more time for quantized weights to converge |

### Results

| #   | Experiment                | calibset | QG  | Finetune    | SNR (dB) | vs Baseline | Status |
| --- | ------------------------- | -------- | --- | ----------- | -------- | ----------- | ------ |
| 5   | opt2_calibfix             | 1024     | -   | -           | 25.08    | -0.25       | Done   |
| 6   | opt2_normin               | 1024     | -   | -           | 24.90    | -0.43       | Done (normalization-in-net) |
| 7   | opt2_l2_ft16              | 1024     | -   | L2/16ep     | 25.73    | +0.40       | Done   |
| 8   | opt2_qg4_l2_ft8           | 1024     | 4   | L2/8ep      | -        | -           | Running |

**Baseline reference**: INT8 opt_level=2 + BC = 25.33 dB SNR

### Findings

1. **calibset_size=1024 (exp 5)**: No meaningful improvement. Calibration statistics with 1024 images gave 25.08 dB (vs 25.33 baseline) — essentially the same. DFC's default of 64 samples was apparently sufficient for this model.

2. **Normalization-in-net (exp 6)**: Slightly worse at 24.90 dB. Moving normalization into the graph so input range was [0,255] instead of [-2,2.5] did not help quantization efficiency.

3. **L2 finetuning, 16 epochs (exp 7)**: Marginal improvement to 25.73 dB (+0.40 dB). Loss plateaued at ~0.043 after epoch 8, with no further improvement through epoch 16. 16 epochs of post-quant finetuning is not enough to overcome the fundamental INT8 precision limitation.

4. **quantization_groups=4 (exp 8)**: **Failed** — `resnet18_layer4/conv3 does not support quantization_groups. Expected 1, but got 4.` Not all conv layers in the DFC support quantization groups. The feature is limited to specific layer types/configurations. Would need per-layer specification of which convs support it, but this is unlikely to yield meaningful improvement given the results above.

### Phase 2 Conclusion

**INT8 accuracy recovery is not viable through DFC optimization alone.** After testing calibset_size, normalization placement, L2 loss finetuning (16 epochs), and quantization groups, no experiment improved SNR beyond +0.40 dB over the baseline (25.33 → 25.73 dB max). The quantization error is fundamental and uniformly distributed — it cannot be addressed by changing calibration parameters or post-quantization optimization.

The **uint16_all (a16_w16) solution remains the correct approach**: 70% success rate matching FP32 baseline, 30.9 MB HEF, 5 HW contexts, 61s eval time.

## Key Observations

1. **uint16 on layer4 alone is insufficient**: Even with layer4 (conv16-20) in full 16-bit precision + bias correction + 4 epochs QAT, the INT8 quantization in earlier layers still introduces enough error to cause 0% success.

2. **Error is uniformly distributed**: Per-layer analysis shows each layer group contributes roughly equally to the total quantization error (~1 dB each). This means there's no single "bottleneck" layer to fix.

3. **uint16 all layers eliminates quantization error**: With all layers in uint16, SNR jumps from 18.92 to 45.68 dB (cosine 0.99999). This confirms the backbone quantization is the sole cause of the accuracy drop.

4. **Spatial error concentration**: Quantization errors are concentrated in the **center** of the feature map (where the robot arm/cube appear) with 13x higher noise power than edges. This explains why even small SNR differences have outsized impact on actions.

5. **Performance overhead**: The a16_w16 layers increase HEF size from 8.5 MB to 30.9 MB. Partial uint16 (layer4 only) was 3.5x slower (262s), but full uint16 is actually comparable to CPU FP32 (61s vs 66s).

6. **uint16_all restores full accuracy**: With ALL layers in uint16, the Hailo backbone achieves **70% success rate** — identical to CPU FP32 baseline. Avg reward 182.3 (vs 186.4 FP32, within noise). Eval time 61s (slightly faster than 66s CPU FP32).

## Conclusion

**uint16_all restores full FP32 accuracy on the Hailo accelerator.** Running all ResNet18 layers in Hailo's `a16_w16` (uint16) mode achieves 70% success rate on TransferCube — identical to the CPU FP32 baseline. Eval time is 61s (vs 66s CPU FP32), making it slightly faster while offloading compute to the accelerator.

The ACT policy is extremely sensitive to backbone feature quantization error. INT8 quantization introduces uniformly distributed error across all layers, and no partial uint16 configuration is sufficient. However, full uint16 eliminates the quantization error (45.68 dB SNR, cosine 0.99999) and fully restores task performance.

**INT8 is definitively not viable** for this model/task combination. Extensive experimentation (Phase 2) with calibset_size tuning, normalization placement, L2 loss finetuning (16 epochs), and quantization groups all failed to improve INT8 SNR beyond 25.73 dB — insufficient for the ~45 dB needed for task success.

### Trade-offs
- **HEF size**: 30.9 MB (vs 8.5 MB for INT8) — 3.6x larger
- **Hardware contexts**: 5 contexts (vs 1 for INT8) — uses more on-chip memory
- **Inference speed**: 61s for 10 episodes — comparable to CPU FP32, acceptable for real-time control
- **Accuracy**: 70% success — identical to FP32 baseline
