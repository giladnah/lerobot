# Layer Noise Analysis Report

**HAR**: `scripts/hailo/artifacts/resnet18_layer4_quantized.har`

**SNR Threshold**: 25.0 dB

## Per-Layer SNR

DFC's `runner.analyze_noise()` returned `None` after ~43 minutes of execution.
Per-layer results are only accessible via the Hailo Model Profiler GUI, not programmatically.

## Alternative: Manual Per-Layer uint16 Analysis

Instead of relying on `analyze_noise()`, we tested each ResNet18 layer group individually
in uint16 (a16_w16) with opt_level=0. Results are documented in
[report_experiments.md](report_experiments.md#per-layer-uint16-noise-analysis).

| Config | SNR (dB) | Cosine Similarity |
|--------|----------|-------------------|
| baseline INT8 (all layers) | 18.92 | 0.9937 |
| uint16 on layer1 only | 20.04 | 0.9951 |
| uint16 on layer2 only | 19.90 | 0.9950 |
| uint16 on layer3 only | 19.64 | 0.9946 |
| uint16 on layer4 only | 21.09 | 0.9961 |
| **uint16 on ALL layers** | **45.68** | **0.99999** |

**Conclusion**: Error is uniformly distributed across all layers. No single layer is a bottleneck.
