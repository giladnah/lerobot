# ACT Inference Benchmark Report: Hailo vs CPU

## Executive Summary

This report profiles the ACT (Action Chunking Transformer) inference pipeline to understand where time is spent and whether the Hailo-10H accelerator provides a latency advantage. The key finding: **the Hailo-10H in uint16 mode is 1.3x slower than CPU for backbone inference (108 ms vs 76 ms)**, and the MuJoCo simulation's `env.step()` dominates real eval time at 74-78% of the loop. In real-world robot deployment where `env.step()` does not exist, Hailo's value lies in freeing the CPU for other tasks (sensor fusion, planning) rather than in raw latency reduction.

## What is `env.step()`?

This is a critical distinction for interpreting the benchmark results.

### In simulation (gym_aloha / MuJoCo)

`env.step(action)` is the **simulation engine call**. It does three things that do not exist in real-world robot control:

1. **MuJoCo physics simulation** (`physics.step(n_sub_steps)`) — advances the simulated world by one control timestep (0.02s at 50 Hz), computing rigid-body dynamics, contact forces, joint constraints
2. **Image rendering** (`physics.render(480, 640, camera_id="top")`) — rasterizes the simulated scene into a 480x640 pixel image. This is the most expensive part: it runs the MuJoCo renderer headless via EGL
3. **Reward computation** — checks contact pairs (cube touching target) to determine if the task succeeded

None of these exist in real-world deployment. `env.step()` is purely simulation infrastructure.

### In real-world robot control

There is no `env.step()`. The equivalent loop is:

1. **Read sensors** (~1-5 ms) — read joint positions from motor controllers via USB/CAN bus
2. **Capture camera frame** (~33 ms at 30 fps) — grab image from USB/RealSense camera
3. **Send motor commands** (~1-5 ms) — write target positions to motor controllers

The timing is fundamentally different: it is governed by hardware I/O latency and camera frame rate, not physics computation. The real bottleneck on a physical robot is the **camera frame rate** (typically 30 Hz = 33 ms per frame) and the **policy inference time** (~140 ms for a full model call).

### Implication

The 74-78% share of `env.step()` in our benchmarks is **simulation-only overhead**. On a real robot, the entire eval loop looks completely different: the policy inference (~140 ms) is the dominant cost, and the question becomes whether the Hailo accelerator can reduce that.

## Accuracy Results

Task: **TransferCube** (Aloha sim), 10 episodes, seed 1000, 400 steps/episode.

### By backbone type (n_action_steps=100)

| Configuration                            | Success Rate | Avg Reward | Eval Time | Notes                           |
| ---------------------------------------- | ------------ | ---------- | --------- | ------------------------------- |
| **CPU FP32**                             | **70%**      | 186.4      | 66s       | Baseline                        |
| **Hailo uint16_all**                     | **70%**      | 182.3      | 61s       | Matches baseline                |
| Hailo INT8 opt_level=1                   | 0%           | 0.0        | 81s       | Quantization destroys accuracy  |
| Hailo INT8 opt_level=2 + bias correction | 0%           | 0.0        | 112s      | Still 0% — INT8 is insufficient |
| Hailo uint16_layer4 only                 | 0%           | 0.0        | 262s      | Partial uint16 insufficient     |

### By n_action_steps (CPU FP32)

| n_action_steps                    | Success Rate | Avg Reward | Eval Time | Model calls/ep |
| --------------------------------- | ------------ | ---------- | --------- | -------------- |
| 1 (temporal ensemble, coeff=0.01) | **60%**      | 164.8      | 675s      | 400            |
| 5                                 | **0%**       | 48.2       | 223s      | 80             |
| 10                                | **10%**      | 54.6       | 139s      | 40             |
| 20                                | **50%**      | 166.3      | 96s       | 20             |
| 50                                | **70%**      | 185.4      | 67s       | 8              |
| **100** (default)                 | **70%**      | 186.4      | 66s       | 4              |

**Key accuracy findings**:

- Only full uint16 (a16_w16 on ALL layers) restores FP32-level accuracy. Any INT8 quantization results in 0% success.
- n_action_steps >= 50 is required for full accuracy (70%). Lower values cause action discontinuities between chunks.
- Temporal ensembling (n_action_steps=1) partially recovers accuracy (60%) by smoothing across chunks, but at 10x the inference cost (675s vs 66s).
- n*action_steps=5-10 is \_worse* than n_action_steps=1: frequent re-planning without smoothing causes jerky, unstable motion.

## Component Profile

Single `ACT.forward()` call decomposed into timed components (30 iterations, 5 warmup):

| Component                          | CPU (ms)  | CPU % | Hailo (ms) | Hailo % |
| ---------------------------------- | --------- | ----- | ---------- | ------- |
| **Backbone** (ResNet18)            | 76.1      | 57.0% | 110.2      | 67.9%   |
| **Encoder** (4 transformer layers) | 42.3      | 31.7% | 44.5       | 27.4%   |
| **Decoder** (1 transformer layer)  | 5.8       | 4.4%  | 7.2        | 4.4%    |
| Projections                        | 1.7       | 1.3%  | 2.3        | 1.4%    |
| Action head                        | 0.03      | 0.0%  | 0.3        | 0.2%    |
| **Full forward**                   | **133.5** | 100%  | **162.2**  | 100%    |

The backbone is the single largest component in both cases. However, the Hailo backbone (110 ms) is **1.45x slower** than the CPU backbone (76 ms). The encoder/decoder timings are similar since they always run on CPU.

## Backbone Isolation

Detailed breakdown of Hailo backbone overhead (30 iterations):

| Stage                  | Time (ms) | Notes                           |
| ---------------------- | --------- | ------------------------------- |
| **CPU FP32 ResNet18**  | **82.3**  | PyTorch, single-threaded CPU    |
|                        |           |                                 |
| **Hailo full forward** | **108.8** | Total including all conversions |
| NCHW->NHWC + numpy     | 0.18      | Negligible                      |
| create_bindings        | 0.03      | Negligible                      |
| **run()**              | **107.1** | 98.4% of Hailo time             |
| NHWC->NCHW torch       | 0.01      | Negligible                      |

**Speedup: 0.76x** (Hailo is slower).

The overhead is not in data conversion (0.22 ms total) — it is in the Hailo hardware execution itself (107 ms). The uint16 (a16_w16) mode uses 5 hardware contexts and 30.9 MB HEF, which is significantly larger than INT8 (8.5 MB, 1 context). This added precision comes at a computational cost on the Hailo-10H.

## End-to-End Inference

`policy.select_action(batch)` timing — first call (model runs) vs subsequent calls (queue pop):

| Metric                             | CPU       | Hailo     |
| ---------------------------------- | --------- | --------- |
| Model call                         | 140.1 ms  | 160.3 ms  |
| Queue pop                          | 0.0004 ms | 0.0006 ms |
| Max freq (model call only)         | 7.1 Hz    | 6.2 Hz    |
| Avg freq (with n_action_steps=100) | 713.8 Hz  | 623.6 Hz  |

With the default n_action_steps=100, the model runs only 4 times per 400-step episode. The remaining 396 steps are instant queue pops. This makes the backbone speed nearly irrelevant at this setting.

## n_action_steps Sweep — Accuracy and Latency

The `n_action_steps` parameter controls how many steps the policy executes from each predicted chunk before re-running the model. Larger values mean fewer model calls but the actions become increasingly stale. This is the most important hyperparameter for balancing accuracy vs inference cost.

### Accuracy (CPU FP32, 10 episodes, TransferCube)

| n_action_steps        | Success Rate | Avg Reward | Model calls/ep | Eval Time | Notes                        |
| --------------------- | ------------ | ---------- | -------------- | --------- | ---------------------------- |
| 1 (temporal ensemble) | **60%**      | 164.8      | 400            | 675s      | coeff=0.01, model every step |
| 5                     | **0%**       | 48.2       | 80             | 223s      | Too-frequent re-planning     |
| 10                    | **10%**      | 54.6       | 40             | 139s      | Borderline — mostly fails    |
| 20                    | **50%**      | 166.3      | 20             | 96s       | Partial success              |
| 50                    | **70%**      | 185.4      | 8              | 67s       | Matches baseline             |
| **100** (default)     | **70%**      | 186.4      | 4              | 66s       | Best — full chunk used       |

**Key insight**: The accuracy curve is non-monotonic. n*action_steps=5 and n_action_steps=10 are \_worse* than n_action_steps=1 (temporal ensemble). This is because:

- At n_action_steps=1 with temporal ensembling, the model runs every step but outputs are smoothed — the ensemble averages across overlapping chunks, providing stable control.
- At n*action_steps=5-10 \_without* temporal ensembling, the model runs frequently but each time it produces a fresh chunk that abruptly replaces the previous one. These discontinuities between chunks cause jerky motion and task failure.
- At n_action_steps=50-100, each chunk plays out almost completely before re-planning, so transitions between chunks are infrequent and the policy behaves as intended.

### Latency projections (model-only time per episode)

| n_action_steps        | Model calls | CPU total | Hailo total | CPU ms/step | Hailo ms/step | 50 Hz feasible? |
| --------------------- | ----------- | --------- | ----------- | ----------- | ------------- | --------------- |
| 1 (temporal ensemble) | 400         | 57.8s     | 65.8s       | 144.4       | 164.5         | NO              |
| 5                     | 80          | 11.6s     | 13.2s       | 28.9        | 32.9          | NO              |
| **10**                | 40          | 5.8s      | 6.6s        | 14.4        | 16.5          | YES             |
| 20                    | 20          | 2.9s      | 3.3s        | 7.2         | 8.2           | YES             |
| 50                    | 8           | 1.2s      | 1.3s        | 2.9         | 3.3           | YES             |
| **100** (default)     | 4           | 0.6s      | 0.7s        | 1.4         | 1.6           | YES             |

The 50 Hz threshold is 20 ms per step. Both CPU and Hailo meet this at n_action_steps >= 10. However, accuracy collapses below n_action_steps=50, so the feasibility question is moot for those settings.

**The practical operating point is n_action_steps=50 or 100**, where accuracy is at the 70% maximum and inference cost is minimal (4-8 model calls per episode).

## Environment Overhead Decomposition

Full rollout loop with MuJoCo simulation (400 steps, n_action_steps=100):

| Component             | CPU ms/step   | CPU %     | Hailo ms/step | Hailo %   |
| --------------------- | ------------- | --------- | ------------- | --------- |
| **env.step()**        | **7.8**       | **75.2%** | **9.0**       | **77.0%** |
| preprocess_obs        | 0.7           | 6.7%      | 0.7           | 6.2%      |
| select_action (avg)   | 1.9           | 18.0%     | 1.9           | 16.7%     |
| model calls (n=4)     | 136.9 ms each |           | 158.0 ms each |           |
| queue pops (n=396)    | 0.50 ms each  |           | 0.37 ms each  |           |
| framework overhead    | 0.01          | 0.1%      | 0.01          | 0.1%      |
| **Episode wall time** | **4.1s**      |           | **4.7s**      |           |

`env.step()` (MuJoCo physics + rendering) consumes 75-77% of the loop. The policy inference averages only 1.9 ms/step because the model runs 4 times (at ~140-160 ms) while the other 396 steps are ~0.4 ms queue pops.

## Hailo Optimization Tests

| Optimization                         | Backbone (ms) | Speedup |
| ------------------------------------ | ------------- | ------- |
| Baseline (per-call alloc)            | 108.6         | 1.00x   |
| Buffer reuse (pre-allocate bindings) | 107.2         | 1.01x   |
| run_async()                          | 107.2         | 1.01x   |

Buffer reuse and async provide negligible improvement — the bottleneck is the hardware execution itself, not Python-level overhead.

### Pipelining Estimate

Since Hailo runs on dedicated hardware, it can theoretically execute in parallel with CPU work:

| Execution                   | Time (ms) | Notes                              |
| --------------------------- | --------- | ---------------------------------- |
| Hailo backbone              | 107.2     | On Hailo NPU                       |
| Encoder + decoder (CPU)     | 50.3      | On CPU                             |
| **Sequential** (current)    | **157.5** | Sum                                |
| **Pipelined** (theoretical) | **107.2** | Max of the two — **1.47x speedup** |

If async pipelining were implemented (start Hailo backbone, run encoder/decoder on CPU from previous frame's features, collect Hailo result), the end-to-end latency could drop from 157 ms to 107 ms. This would require a one-frame delay on backbone features.

## Summary Table

| Scenario                        | CPU     | Hailo uint16    | Hailo advantage?                        |
| ------------------------------- | ------- | --------------- | --------------------------------------- |
| **Accuracy** (TransferCube)     | 70%     | 70%             | Parity                                  |
| Backbone latency                | 76 ms   | 108 ms          | **No** — 1.4x slower                    |
| Full forward latency            | 133 ms  | 162 ms          | **No** — 1.2x slower                    |
| Eval time (n_action_steps=100)  | 4.1s/ep | 4.7s/ep         | **No** — sim-dominated                  |
| CPU freed during backbone       | 0%      | ~67% of forward | **Yes** — CPU available for other tasks |
| Pipelined latency (theoretical) | 133 ms  | 107 ms          | **Yes** — 1.24x faster                  |
| HEF size                        | N/A     | 30.9 MB         | 5 HW contexts used                      |

## Conclusions

1. **Hailo uint16 matches CPU FP32 accuracy** (70% success rate on TransferCube). INT8 quantization at any optimization level is insufficient (0% success).
2. **Hailo is not faster than CPU for raw backbone latency.** The uint16 mode on Hailo-10H takes 108 ms vs 76 ms for CPU FP32 ResNet18. The precision requirement (a16_w16) eliminates the throughput advantage that INT8 would provide.
3. **Simulation eval times are dominated by MuJoCo** (75-77% of the loop is `env.step()`). This overhead does not exist in real-world deployment. The 61s vs 66s difference in prior eval runs is noise within the simulation overhead.
4. **Hailo's real value is CPU offloading, not latency reduction.** On a real robot, the CPU is busy with: camera capture, sensor fusion, motor control, safety monitoring, and planning. Offloading the backbone (67% of the forward pass) to dedicated hardware frees CPU cycles for these concurrent tasks.
5. **Pipelining could provide real speedup.** Running the Hailo backbone in parallel with the CPU encoder/decoder (using features from the previous frame) could reduce end-to-end latency from 157 ms to 107 ms — a genuine 1.47x improvement. This would require a one-frame feature delay but is architecturally straightforward.
6. **For real-time 50 Hz control**, n_action_steps >= 10 is feasible with either CPU or Hailo. At the default n_action_steps=100, the model runs only 4 times per episode and latency is irrelevant.
