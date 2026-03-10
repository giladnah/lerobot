"""Benchmark ACT inference pipeline — Hailo vs CPU.

Profiles per-component latency, backbone isolation timings, end-to-end inference,
n_action_steps sweep, environment overhead decomposition, and Hailo optimizations
(buffer reuse, async pipelining).

Usage:
    # CPU-only component profile (no Hailo or env needed):
    .venv/bin/python scripts/hailo/benchmark_inference.py --mode component_profile

    # Backbone isolation (needs Hailo):
    .venv/bin/python scripts/hailo/benchmark_inference.py --mode backbone

    # End-to-end inference latency:
    .venv/bin/python scripts/hailo/benchmark_inference.py --mode e2e

    # n_action_steps sweep (needs env):
    MUJOCO_GL=egl .venv/bin/python scripts/hailo/benchmark_inference.py --mode sweep

    # Environment overhead decomposition (needs env):
    MUJOCO_GL=egl .venv/bin/python scripts/hailo/benchmark_inference.py --mode env

    # Hailo optimization tests (buffer reuse, async):
    .venv/bin/python scripts/hailo/benchmark_inference.py --mode optimize

    # All modes:
    MUJOCO_GL=egl .venv/bin/python scripts/hailo/benchmark_inference.py --mode all
"""

import argparse
import json
import time
from pathlib import Path

import einops
import numpy as np
import torch

torch.set_grad_enabled(False)

# Default paths
DEFAULT_CPU_PATH = "outputs/migrated/act_aloha_sim_transfer_cube_human"
DEFAULT_HAILO_PATH = "outputs/migrated/act_aloha_sim_transfer_cube_human_hailo_uint16_all"
DEFAULT_HEF = "scripts/hailo/artifacts/resnet18_layer4_opt2_uint16_all.hef"


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


class Timer:
    """Context-manager timer using perf_counter."""

    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def time_fn(fn, n_warmup=3, n_iter=20):
    """Time a callable over n_iter iterations after warmup. Returns (mean_ms, std_ms, all_ms)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iter):
        t = Timer()
        with t:
            fn()
        times.append(t.elapsed_ms)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std()), times


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_policy(path, device="cpu"):
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(path)
    policy.to(device)
    policy.eval()
    return policy


def make_env():
    import gym_aloha  # noqa: F401
    import gymnasium as gym

    return gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")


def make_dummy_batch(device="cpu"):
    """Create a realistic dummy batch matching Aloha sim observations."""
    return {
        "observation.images.top": torch.randn(1, 3, 480, 640, device=device),
        "observation.state": torch.randn(1, 14, device=device),
    }


def prepare_model_batch(policy, batch):
    """Convert a select_action-style batch into the format ACT.forward() expects."""
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    model_batch = {}
    model_batch[OBS_STATE] = batch["observation.state"]
    model_batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]
    return model_batch


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def print_table(headers, rows, title=None):
    """Print a formatted ASCII table."""
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")

    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2 for i, h in enumerate(headers)
    ]
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths, strict=False))
    print(f"  {header_line}")
    print(f"  {'-' * len(header_line)}")
    for row in rows:
        line = "".join(str(v).ljust(w) for v, w in zip(row, col_widths, strict=False))
        print(f"  {line}")
    print()


# ---------------------------------------------------------------------------
# Mode 1: Component Profile
# ---------------------------------------------------------------------------


def run_component_profile(policy, batch, n_warmup, n_iter, label="CPU"):
    """Decompose ACT.forward() into timed components."""
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    model = policy.model
    model_batch = prepare_model_batch(policy, batch)
    device = batch["observation.state"].device

    results = {}

    # 1. Backbone
    img = model_batch[OBS_IMAGES][0]
    mean, std, _ = time_fn(lambda: model.backbone(img), n_warmup, n_iter)
    results["backbone"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    # Get backbone output for subsequent stages
    cam_features = model.backbone(img)["feature_map"]

    # 2. Projections (image feat proj + pos embed + latent proj + state proj)
    def run_projections():
        batch_size = img.shape[0]
        latent_sample = torch.zeros([batch_size, model.config.latent_dim], dtype=torch.float32, device=device)
        tokens = [model.encoder_latent_input_proj(latent_sample)]
        pos_embeds = list(model.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if model.config.robot_state_feature:
            tokens.append(model.encoder_robot_state_input_proj(model_batch[OBS_STATE]))
        cam_proj = model.encoder_img_feat_input_proj(cam_features)
        cam_pos = model.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
        cam_proj_r = einops.rearrange(cam_proj, "b c h w -> (h w) b c")
        cam_pos_r = einops.rearrange(cam_pos, "b c h w -> (h w) b c")
        tokens.extend(list(cam_proj_r))
        pos_embeds.extend(list(cam_pos_r))
        return torch.stack(tokens, axis=0), torch.stack(pos_embeds, axis=0)

    mean, std, _ = time_fn(run_projections, n_warmup, n_iter)
    results["projections"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    # Get projection outputs
    encoder_in_tokens, encoder_in_pos_embed = run_projections()

    # 3. Encoder (4 transformer layers)
    mean, std, _ = time_fn(
        lambda: model.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed), n_warmup, n_iter
    )
    results["encoder"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    encoder_out = model.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

    # 4. Decoder (1 transformer layer)
    batch_size = img.shape[0]
    decoder_in = torch.zeros(
        (model.config.chunk_size, batch_size, model.config.dim_model),
        dtype=encoder_in_pos_embed.dtype,
        device=device,
    )

    def run_decoder():
        return model.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=model.decoder_pos_embed.weight.unsqueeze(1),
        )

    mean, std, _ = time_fn(run_decoder, n_warmup, n_iter)
    results["decoder"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    decoder_out = run_decoder().transpose(0, 1)

    # 5. Action head
    mean, std, _ = time_fn(lambda: model.action_head(decoder_out), n_warmup, n_iter)
    results["action_head"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    # 6. Full forward for comparison
    mean, std, _ = time_fn(lambda: model(model_batch), n_warmup, n_iter)
    results["full_forward"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    # Print table
    total = results["full_forward"]["mean_ms"]
    rows = []
    for name in ["backbone", "projections", "encoder", "decoder", "action_head"]:
        m = results[name]["mean_ms"]
        s = results[name]["std_ms"]
        pct = (m / total * 100) if total > 0 else 0
        rows.append([name, f"{m:.2f}", f"{s:.2f}", f"{pct:.1f}%"])
    rows.append(["---", "---", "---", "---"])
    rows.append(["full_forward", f"{total:.2f}", f"{results['full_forward']['std_ms']:.2f}", "100.0%"])
    sum_parts = sum(
        results[k]["mean_ms"] for k in ["backbone", "projections", "encoder", "decoder", "action_head"]
    )
    rows.append(["sum_of_parts", f"{sum_parts:.2f}", "-", f"{sum_parts / total * 100:.1f}%"])

    print_table(["Component", "Mean (ms)", "Std (ms)", "% of total"], rows, f"Component Profile [{label}]")
    return results


# ---------------------------------------------------------------------------
# Mode 2: Backbone Isolation
# ---------------------------------------------------------------------------


def run_backbone_isolation(cpu_policy, hailo_policy, hef_path, batch, n_warmup, n_iter):
    """Compare backbone call timings: CPU FP32 vs Hailo with sub-timings."""
    results = {}
    img = batch["observation.images.top"]  # (1, 3, 480, 640)

    # CPU backbone
    cpu_backbone = cpu_policy.model.backbone
    mean, std, all_times = time_fn(lambda: cpu_backbone(img), n_warmup, n_iter)
    results["cpu_backbone"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    # Hailo backbone (full forward including conversions)
    hailo_backbone = hailo_policy.model.backbone
    mean, std, all_times = time_fn(lambda: hailo_backbone(img), n_warmup, n_iter)
    results["hailo_full"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    # Hailo sub-timings: break down the forward pass
    x_nhwc = img.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy()
    sample = np.ascontiguousarray(x_nhwc[0])

    # Data conversion (NCHW->NHWC + numpy)
    def data_to_nhwc():
        return img.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy()

    mean, std, _ = time_fn(data_to_nhwc, n_warmup, n_iter)
    results["hailo_data_conversion"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    # Create bindings + set buffers
    configured_model = hailo_backbone._configured_model
    output_shape = hailo_backbone._output_shape

    def create_and_bind():
        bindings = configured_model.create_bindings()
        bindings.input().set_buffer(sample)
        out_buf = np.empty(output_shape, dtype=np.float32)
        bindings.output().set_buffer(out_buf)
        return bindings

    mean, std, _ = time_fn(create_and_bind, n_warmup, n_iter)
    results["hailo_create_bindings"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    # run() only
    bindings = create_and_bind()

    def hailo_run_only():
        configured_model.run([bindings], timeout=10000)

    mean, std, _ = time_fn(hailo_run_only, n_warmup, n_iter)
    results["hailo_run"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    # Output conversion (NHWC numpy -> NCHW torch)
    out_nhwc = np.empty((1,) + tuple(output_shape), dtype=np.float32)

    def output_to_tensor():
        return torch.from_numpy(out_nhwc).permute(0, 3, 1, 2)

    mean, std, _ = time_fn(output_to_tensor, n_warmup, n_iter)
    results["hailo_output_conversion"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    # Print
    rows = [
        [
            "CPU FP32 backbone",
            f"{results['cpu_backbone']['mean_ms']:.2f}",
            f"{results['cpu_backbone']['std_ms']:.2f}",
        ],
        ["", "", ""],
        [
            "Hailo full forward",
            f"{results['hailo_full']['mean_ms']:.2f}",
            f"{results['hailo_full']['std_ms']:.2f}",
        ],
        [
            "  data conversion",
            f"{results['hailo_data_conversion']['mean_ms']:.2f}",
            f"{results['hailo_data_conversion']['std_ms']:.2f}",
        ],
        [
            "  create_bindings",
            f"{results['hailo_create_bindings']['mean_ms']:.2f}",
            f"{results['hailo_create_bindings']['std_ms']:.2f}",
        ],
        ["  run()", f"{results['hailo_run']['mean_ms']:.2f}", f"{results['hailo_run']['std_ms']:.2f}"],
        [
            "  output conversion",
            f"{results['hailo_output_conversion']['mean_ms']:.2f}",
            f"{results['hailo_output_conversion']['std_ms']:.2f}",
        ],
    ]
    sub_sum = sum(
        results[k]["mean_ms"]
        for k in ["hailo_data_conversion", "hailo_create_bindings", "hailo_run", "hailo_output_conversion"]
    )
    rows.append(["  sum of sub-parts", f"{sub_sum:.2f}", "-"])

    speedup = (
        results["cpu_backbone"]["mean_ms"] / results["hailo_full"]["mean_ms"]
        if results["hailo_full"]["mean_ms"] > 0
        else 0
    )
    rows.append(["", "", ""])
    rows.append(["Speedup (CPU/Hailo)", f"{speedup:.2f}x", ""])

    print_table(["Stage", "Mean (ms)", "Std (ms)"], rows, "Backbone Isolation")
    return results


# ---------------------------------------------------------------------------
# Mode 3: End-to-End Inference (no env)
# ---------------------------------------------------------------------------


def run_e2e_inference(policy, batch, n_warmup, n_iter, label="CPU"):
    """Time policy.select_action() — first call vs queue pops."""
    results = {}

    # First call (model runs)
    def first_call():
        policy.reset()
        return policy.select_action(batch)

    mean, std, _ = time_fn(first_call, n_warmup, n_iter)
    results["first_call"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    # Subsequent calls (queue pop)
    policy.reset()
    policy.select_action(batch)  # fill queue

    def queue_pop():
        if len(policy._action_queue) == 0:
            policy.reset()
            policy.select_action(batch)
        return policy._action_queue.popleft()

    # We need to measure pure popleft time
    policy.reset()
    policy.select_action(batch)
    pop_times = []
    for _ in range(min(n_iter, policy.config.n_action_steps - 1)):
        if len(policy._action_queue) == 0:
            break
        t = Timer()
        with t:
            policy._action_queue.popleft()
        pop_times.append(t.elapsed_ms)

    if pop_times:
        arr = np.array(pop_times)
        results["queue_pop"] = {"mean_ms": round(float(arr.mean()), 4), "std_ms": round(float(arr.std()), 4)}
    else:
        results["queue_pop"] = {"mean_ms": 0.0, "std_ms": 0.0}

    # Max control frequency
    model_call_ms = results["first_call"]["mean_ms"]
    pop_ms = results["queue_pop"]["mean_ms"]
    n_action_steps = policy.config.n_action_steps
    # Average time per action step over a chunk
    avg_per_step_ms = (model_call_ms + pop_ms * (n_action_steps - 1)) / n_action_steps
    max_freq = 1000.0 / model_call_ms if model_call_ms > 0 else float("inf")
    avg_freq = 1000.0 / avg_per_step_ms if avg_per_step_ms > 0 else float("inf")

    results["max_freq_model_call_hz"] = round(max_freq, 1)
    results["avg_freq_with_chunking_hz"] = round(avg_freq, 1)
    results["n_action_steps"] = n_action_steps

    rows = [
        ["select_action (model call)", f"{model_call_ms:.2f}", f"{results['first_call']['std_ms']:.2f}"],
        ["select_action (queue pop)", f"{pop_ms:.4f}", f"{results['queue_pop']['std_ms']:.4f}"],
        ["", "", ""],
        [f"n_action_steps={n_action_steps}", "", ""],
        ["Max freq (model call)", f"{max_freq:.1f} Hz", ""],
        ["Avg freq (with chunking)", f"{avg_freq:.1f} Hz", ""],
    ]
    print_table(["Metric", "Value", "Std"], rows, f"End-to-End Inference [{label}]")
    return results


# ---------------------------------------------------------------------------
# Mode 4: n_action_steps Sweep
# ---------------------------------------------------------------------------


def run_sweep(policy, batch, n_warmup, n_iter, label="CPU"):
    """Simulate 400-step episodes with varying n_action_steps."""
    results = {}
    episode_length = 400

    # Measure single model call time
    def model_call():
        policy.reset()
        return policy.select_action(batch)

    model_mean, model_std, _ = time_fn(model_call, n_warmup, n_iter)

    sweep_values = [1, 5, 10, 20, 50, 100]
    rows = []

    for n_steps in sweep_values:
        n_model_calls = episode_length if n_steps == 1 else (episode_length + n_steps - 1) // n_steps

        total_model_ms = n_model_calls * model_mean
        total_model_s = total_model_ms / 1000
        # Time available at 50 Hz: 20ms per step, 400 steps = 8s total
        budget_at_50hz_ms = episode_length * 20  # 8000 ms
        feasible = "YES" if total_model_ms < budget_at_50hz_ms else "NO"
        ms_per_step = total_model_ms / episode_length

        entry = {
            "n_action_steps": n_steps,
            "n_model_calls": n_model_calls,
            "total_model_ms": round(total_model_ms, 1),
            "ms_per_step": round(ms_per_step, 2),
            "feasible_50hz": feasible,
        }
        results[f"n_steps_{n_steps}"] = entry
        rows.append(
            [
                n_steps,
                n_model_calls,
                f"{total_model_s:.2f}s",
                f"{ms_per_step:.2f}",
                feasible,
            ]
        )

    results["model_call_mean_ms"] = round(model_mean, 2)
    results["model_call_std_ms"] = round(model_std, 2)

    print(f"\n  Model call latency: {model_mean:.2f} +/- {model_std:.2f} ms")
    print_table(
        ["n_action_steps", "Model calls/ep", "Total model time", "ms/step (avg)", "50 Hz feasible?"],
        rows,
        f"n_action_steps Sweep (400-step episode) [{label}]",
    )
    return results


# ---------------------------------------------------------------------------
# Mode 5: Env Overhead Decomposition
# ---------------------------------------------------------------------------


def run_env_decomposition(policy, n_episodes=2, label="CPU"):
    """Run episodes timing each component of the rollout loop."""
    env = make_env()
    episode_length = 400
    results = {"episodes": []}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        policy.reset()

        times_env_step = []
        times_preprocess = []
        times_select_action = []
        times_loop = []
        model_call_count = 0

        for _step in range(episode_length):
            t_loop = Timer()
            with t_loop:
                # Preprocess observation
                t_pre = Timer()
                with t_pre:
                    obs_dict = {
                        "observation.images.top": torch.from_numpy(obs["pixels"]["top"])
                        .permute(2, 0, 1)
                        .float()
                        .div(255.0)
                        .unsqueeze(0),
                        "observation.state": torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0),
                    }

                # Check if this will be a model call or queue pop
                is_model_call = not hasattr(policy, "_action_queue") or len(policy._action_queue) == 0

                # Select action
                t_sa = Timer()
                with t_sa:
                    action = policy.select_action(obs_dict)

                if is_model_call:
                    model_call_count += 1

                # Env step
                t_env = Timer()
                with t_env:
                    action_np = action.squeeze(0).numpy()
                    obs, reward, terminated, truncated, info = env.step(action_np)

            times_preprocess.append(t_pre.elapsed_ms)
            times_select_action.append(t_sa.elapsed_ms)
            times_env_step.append(t_env.elapsed_ms)
            times_loop.append(t_loop.elapsed_ms)

        ep_results = {
            "episode": ep,
            "model_calls": model_call_count,
            "env_step": {
                "mean_ms": round(float(np.mean(times_env_step)), 2),
                "std_ms": round(float(np.std(times_env_step)), 2),
            },
            "preprocess": {
                "mean_ms": round(float(np.mean(times_preprocess)), 2),
                "std_ms": round(float(np.std(times_preprocess)), 2),
            },
            "select_action_all": {
                "mean_ms": round(float(np.mean(times_select_action)), 2),
                "std_ms": round(float(np.std(times_select_action)), 2),
            },
            "loop_total": {
                "mean_ms": round(float(np.mean(times_loop)), 2),
                "std_ms": round(float(np.std(times_loop)), 2),
            },
            "episode_total_s": round(sum(times_loop) / 1000, 2),
        }

        # Separate model-call steps from queue-pop steps
        sa_arr = np.array(times_select_action)
        # Model call steps are the ones with highest latency
        threshold = np.median(sa_arr) * 3  # model calls are much slower than pops
        model_mask = sa_arr > threshold
        pop_mask = ~model_mask
        if model_mask.any():
            ep_results["select_action_model"] = {
                "mean_ms": round(float(sa_arr[model_mask].mean()), 2),
                "n": int(model_mask.sum()),
            }
        if pop_mask.any():
            ep_results["select_action_pop"] = {
                "mean_ms": round(float(sa_arr[pop_mask].mean()), 4),
                "n": int(pop_mask.sum()),
            }

        results["episodes"].append(ep_results)

    env.close()

    # Print per-episode summary
    for ep_r in results["episodes"]:
        total = ep_r["loop_total"]["mean_ms"]
        rows = [
            [
                "env.step()",
                f"{ep_r['env_step']['mean_ms']:.2f}",
                f"{ep_r['env_step']['std_ms']:.2f}",
                f"{ep_r['env_step']['mean_ms'] / total * 100:.1f}%",
            ],
            [
                "preprocess_obs",
                f"{ep_r['preprocess']['mean_ms']:.2f}",
                f"{ep_r['preprocess']['std_ms']:.2f}",
                f"{ep_r['preprocess']['mean_ms'] / total * 100:.1f}%",
            ],
            [
                "select_action (all)",
                f"{ep_r['select_action_all']['mean_ms']:.2f}",
                f"{ep_r['select_action_all']['std_ms']:.2f}",
                f"{ep_r['select_action_all']['mean_ms'] / total * 100:.1f}%",
            ],
        ]
        if "select_action_model" in ep_r:
            rows.append(
                [
                    "  model calls",
                    f"{ep_r['select_action_model']['mean_ms']:.2f}",
                    f"n={ep_r['select_action_model']['n']}",
                    "",
                ]
            )
        if "select_action_pop" in ep_r:
            rows.append(
                [
                    "  queue pops",
                    f"{ep_r['select_action_pop']['mean_ms']:.4f}",
                    f"n={ep_r['select_action_pop']['n']}",
                    "",
                ]
            )
        overhead = (
            total
            - ep_r["env_step"]["mean_ms"]
            - ep_r["preprocess"]["mean_ms"]
            - ep_r["select_action_all"]["mean_ms"]
        )
        rows.append(["framework overhead", f"{overhead:.2f}", "-", f"{overhead / total * 100:.1f}%"])
        rows.append(["---", "---", "---", "---"])
        rows.append(["loop total", f"{total:.2f}", f"{ep_r['loop_total']['std_ms']:.2f}", "100.0%"])
        rows.append(["episode wall time", f"{ep_r['episode_total_s']:.2f}s", "", ""])
        rows.append(["model calls", str(ep_r["model_calls"]), "", ""])
        print_table(
            ["Component", "Mean/step (ms)", "Std/Info", "% of loop"],
            rows,
            f"Env Overhead Decomposition [{label}] Episode {ep_r['episode']}",
        )

    return results


# ---------------------------------------------------------------------------
# Mode 6: Optimization Tests
# ---------------------------------------------------------------------------


def run_optimize(hailo_policy, hef_path, batch, n_warmup, n_iter):
    """Test Hailo optimizations: buffer reuse and async pipelining."""
    results = {}
    img = batch["observation.images.top"]
    hailo_backbone = hailo_policy.model.backbone

    # --- Baseline: current implementation (allocates per call) ---
    mean, std, _ = time_fn(lambda: hailo_backbone(img), n_warmup, n_iter)
    results["baseline"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    # --- Buffer reuse: pre-allocate bindings + output buffer ---
    configured_model = hailo_backbone._configured_model
    output_shape = hailo_backbone._output_shape

    # Pre-allocate everything once
    x_nhwc = img.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy()
    sample_buf = np.ascontiguousarray(x_nhwc[0])
    reuse_bindings = configured_model.create_bindings()
    reuse_out_buf = np.empty(output_shape, dtype=np.float32)
    reuse_bindings.input().set_buffer(sample_buf)
    reuse_bindings.output().set_buffer(reuse_out_buf)

    def buffer_reuse_forward():
        # Copy new data into pre-allocated buffer
        np.copyto(sample_buf, x_nhwc[0])
        configured_model.run([reuse_bindings], timeout=10000)
        out = reuse_bindings.output().get_buffer()
        return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)

    mean, std, _ = time_fn(buffer_reuse_forward, n_warmup, n_iter)
    results["buffer_reuse"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}

    # --- Async pipelining: use run_async() ---
    try:
        import threading

        async_out_buf = np.empty(output_shape, dtype=np.float32)
        async_bindings = configured_model.create_bindings()
        async_bindings.input().set_buffer(sample_buf)
        async_bindings.output().set_buffer(async_out_buf)

        def async_forward():
            event = threading.Event()

            def callback(completion_info):
                event.set()

            np.copyto(sample_buf, x_nhwc[0])
            configured_model.run_async([async_bindings], callback)
            event.wait(timeout=10.0)
            return torch.from_numpy(async_bindings.output().get_buffer()).permute(2, 0, 1).unsqueeze(0)

        mean, std, _ = time_fn(async_forward, n_warmup, n_iter)
        results["async_run"] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}
    except (AttributeError, TypeError, ImportError):
        results["async_run"] = {"mean_ms": None, "error": "run_async not available"}

    # --- Pipelining estimate: max(backbone, encoder+decoder) ---
    # Get encoder+decoder time from a quick profile
    model = hailo_policy.model
    model_batch = prepare_model_batch(hailo_policy, batch)

    cam_features = hailo_backbone(img)["feature_map"]
    cam_pos = model.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
    cam_proj = model.encoder_img_feat_input_proj(cam_features)
    cam_proj_r = einops.rearrange(cam_proj, "b c h w -> (h w) b c")
    cam_pos_r = einops.rearrange(cam_pos, "b c h w -> (h w) b c")

    batch_size = img.shape[0]
    device = img.device
    latent_sample = torch.zeros([batch_size, model.config.latent_dim], dtype=torch.float32, device=device)
    tokens = [model.encoder_latent_input_proj(latent_sample)]
    pos_embeds = list(model.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
    if model.config.robot_state_feature:
        tokens.append(model.encoder_robot_state_input_proj(model_batch["observation.state"]))
    tokens.extend(list(cam_proj_r))
    pos_embeds.extend(list(cam_pos_r))
    encoder_in_tokens = torch.stack(tokens, axis=0)
    encoder_in_pos_embed = torch.stack(pos_embeds, axis=0)

    def encoder_decoder():
        enc_out = model.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        dec_in = torch.zeros(
            (model.config.chunk_size, batch_size, model.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=device,
        )
        dec_out = model.decoder(
            dec_in,
            enc_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=model.decoder_pos_embed.weight.unsqueeze(1),
        )
        return model.action_head(dec_out.transpose(0, 1))

    mean_ed, std_ed, _ = time_fn(encoder_decoder, n_warmup, n_iter)
    results["encoder_decoder_cpu"] = {"mean_ms": round(mean_ed, 2), "std_ms": round(std_ed, 2)}

    backbone_ms = results["buffer_reuse"]["mean_ms"]
    sequential_ms = backbone_ms + mean_ed
    pipelined_ms = max(backbone_ms, mean_ed)
    results["pipelining_estimate"] = {
        "sequential_ms": round(sequential_ms, 2),
        "pipelined_ms": round(pipelined_ms, 2),
        "speedup": round(sequential_ms / pipelined_ms, 2) if pipelined_ms > 0 else 0,
    }

    # Print
    rows = [
        [
            "Hailo baseline (per-call alloc)",
            f"{results['baseline']['mean_ms']:.2f}",
            f"{results['baseline']['std_ms']:.2f}",
        ],
        [
            "Hailo buffer reuse",
            f"{results['buffer_reuse']['mean_ms']:.2f}",
            f"{results['buffer_reuse']['std_ms']:.2f}",
        ],
    ]
    if results["async_run"].get("mean_ms") is not None:
        rows.append(
            [
                "Hailo run_async",
                f"{results['async_run']['mean_ms']:.2f}",
                f"{results['async_run']['std_ms']:.2f}",
            ]
        )
    else:
        rows.append(["Hailo run_async", "N/A", results["async_run"].get("error", "")])
    rows.append(["", "", ""])
    rows.append(["Encoder+decoder (CPU)", f"{mean_ed:.2f}", f"{std_ed:.2f}"])
    rows.append(["", "", ""])
    rows.append(["Sequential (backbone+enc+dec)", f"{sequential_ms:.2f}", ""])
    rows.append(
        [
            "Pipelined (max of above)",
            f"{pipelined_ms:.2f}",
            f"{results['pipelining_estimate']['speedup']:.2f}x",
        ]
    )

    baseline_speedup = (
        results["baseline"]["mean_ms"] / results["buffer_reuse"]["mean_ms"]
        if results["buffer_reuse"]["mean_ms"] > 0
        else 0
    )
    rows.append(["", "", ""])
    rows.append(["Buffer reuse speedup", f"{baseline_speedup:.2f}x", ""])

    print_table(["Optimization", "Mean (ms)", "Std/Info"], rows, "Optimization Tests")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ACT inference pipeline — Hailo vs CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="component_profile",
        choices=["component_profile", "backbone", "e2e", "sweep", "env", "optimize", "all"],
        help="Which benchmark mode(s) to run",
    )
    parser.add_argument(
        "--policy-path", type=str, default=DEFAULT_CPU_PATH, help="Path to CPU FP32 policy checkpoint"
    )
    parser.add_argument(
        "--hailo-policy-path", type=str, default=DEFAULT_HAILO_PATH, help="Path to Hailo policy checkpoint"
    )
    parser.add_argument("--hef", type=str, default=DEFAULT_HEF, help="Path to HEF file")
    parser.add_argument("--n-warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--n-iter", type=int, default=50, help="Number of timed iterations")
    parser.add_argument(
        "--output", type=str, default="scripts/hailo/reports/benchmark_results.json", help="JSON output path"
    )
    args = parser.parse_args()

    modes = (
        ["component_profile", "backbone", "e2e", "sweep", "env", "optimize"]
        if args.mode == "all"
        else [args.mode]
    )

    # Check which optional deps are available
    has_hailo = False
    try:
        from hailo_platform import VDevice  # noqa: F401

        has_hailo = True
    except ImportError:
        pass

    has_env = False
    try:
        import gym_aloha  # noqa: F401

        has_env = True
    except ImportError:
        pass

    all_results = {"metadata": {"n_warmup": args.n_warmup, "n_iter": args.n_iter}}

    # Load policies as needed
    cpu_policy = None
    hailo_policy = None

    needs_cpu = any(m in modes for m in ["component_profile", "backbone", "e2e", "sweep", "env"])
    needs_hailo = any(
        m in modes for m in ["backbone", "e2e", "sweep", "optimize", "component_profile", "env"]
    )

    if needs_cpu:
        print(f"Loading CPU policy from {args.policy_path}...")
        cpu_policy = load_policy(args.policy_path)
        print("  Done.")

    if needs_hailo and has_hailo:
        print(f"Loading Hailo policy from {args.hailo_policy_path}...")
        try:
            hailo_policy = load_policy(args.hailo_policy_path)
            print("  Done.")
        except Exception as e:
            print(f"  Failed to load Hailo policy: {e}")
            hailo_policy = None

    batch = make_dummy_batch()

    # ---- Run modes ----

    if "component_profile" in modes:
        print("\n" + "=" * 70)
        print("  MODE: component_profile")
        print("=" * 70)
        if cpu_policy:
            all_results["component_profile_cpu"] = run_component_profile(
                cpu_policy, batch, args.n_warmup, args.n_iter, label="CPU"
            )
        if hailo_policy:
            all_results["component_profile_hailo"] = run_component_profile(
                hailo_policy, batch, args.n_warmup, args.n_iter, label="Hailo"
            )
        elif not cpu_policy:
            print("  SKIPPED: no policy loaded")

    if "backbone" in modes:
        print("\n" + "=" * 70)
        print("  MODE: backbone")
        print("=" * 70)
        if not has_hailo or not hailo_policy:
            print("  SKIPPED: Hailo not available")
        elif not cpu_policy:
            print("  SKIPPED: CPU policy not loaded")
        else:
            all_results["backbone_isolation"] = run_backbone_isolation(
                cpu_policy, hailo_policy, args.hef, batch, args.n_warmup, args.n_iter
            )

    if "e2e" in modes:
        print("\n" + "=" * 70)
        print("  MODE: e2e")
        print("=" * 70)
        if cpu_policy:
            all_results["e2e_cpu"] = run_e2e_inference(
                cpu_policy, batch, args.n_warmup, args.n_iter, label="CPU"
            )
        if hailo_policy:
            all_results["e2e_hailo"] = run_e2e_inference(
                hailo_policy, batch, args.n_warmup, args.n_iter, label="Hailo"
            )

    if "sweep" in modes:
        print("\n" + "=" * 70)
        print("  MODE: sweep")
        print("=" * 70)
        if cpu_policy:
            all_results["sweep_cpu"] = run_sweep(cpu_policy, batch, args.n_warmup, args.n_iter, label="CPU")
        if hailo_policy:
            all_results["sweep_hailo"] = run_sweep(
                hailo_policy, batch, args.n_warmup, args.n_iter, label="Hailo"
            )

    if "env" in modes:
        print("\n" + "=" * 70)
        print("  MODE: env")
        print("=" * 70)
        if not has_env:
            print("  SKIPPED: gym_aloha not available")
        else:
            if cpu_policy:
                all_results["env_cpu"] = run_env_decomposition(cpu_policy, n_episodes=2, label="CPU")
            if hailo_policy:
                all_results["env_hailo"] = run_env_decomposition(hailo_policy, n_episodes=2, label="Hailo")

    if "optimize" in modes:
        print("\n" + "=" * 70)
        print("  MODE: optimize")
        print("=" * 70)
        if not has_hailo or not hailo_policy:
            print("  SKIPPED: Hailo not available")
        else:
            all_results["optimize"] = run_optimize(hailo_policy, args.hef, batch, args.n_warmup, args.n_iter)

    # ---- Save JSON ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
