"""Diagnose Hailo INT8 action errors vs CPU FP32.

Runs both policies on the same environment seed and compares:
1. Per-step backbone feature differences (cosine, bias, magnitude)
2. Per-step action differences (normalized)
3. Whether error is systematic (biased) or random noise

Usage:
    MUJOCO_GL=egl python scripts/hailo/diagnose_hailo_actions.py \
        --cpu-path outputs/migrated/act_aloha_sim_transfer_cube_human \
        --hailo-path outputs/migrated/act_aloha_sim_transfer_cube_human_hailo_opt2
"""

import argparse

import numpy as np
import torch

torch.set_grad_enabled(False)


def load_policy(path, device="cpu"):
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(path)
    policy.to(device)
    policy.eval()
    return policy


def make_env():
    import gym_aloha  # noqa: F401
    import gymnasium as gym

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    return env


def compare_features(cpu_feat, hailo_feat, label=""):
    """Compare two feature tensors and print statistics."""
    cpu_flat = cpu_feat.flatten()
    hailo_flat = hailo_feat.flatten()
    cosine = torch.nn.functional.cosine_similarity(cpu_flat.unsqueeze(0), hailo_flat.unsqueeze(0)).item()
    diff = cpu_flat - hailo_flat
    print(f"  {label} Features:")
    print(f"    Cosine similarity: {cosine:.6f}")
    print(f"    Mean diff (bias): {diff.mean():.6f}")
    print(f"    Mean abs diff: {diff.abs().mean():.6f}")
    print(f"    Max abs diff: {diff.abs().max():.6f}")
    print(f"    Std of diff: {diff.std():.6f}")
    print(f"    CPU feature range: [{cpu_flat.min():.3f}, {cpu_flat.max():.3f}], mean={cpu_flat.mean():.3f}")
    print(
        f"    Hailo feature range: [{hailo_flat.min():.3f}, {hailo_flat.max():.3f}], mean={hailo_flat.mean():.3f}"
    )
    return cosine, diff


def compare_actions(cpu_action, hailo_action, label=""):
    """Compare two action tensors."""
    diff = cpu_action - hailo_action
    print(f"  {label} Actions (14 joints):")
    print(f"    Mean diff (bias): {diff.mean():.6f}")
    print(f"    Mean abs diff: {diff.abs().mean():.6f}")
    print(f"    Max abs diff: {diff.abs().max():.6f}")
    print("    Per-joint:")
    for i in range(len(diff)):
        print(f"      J{i:2d}: cpu={cpu_action[i]:+.6f}  hailo={hailo_action[i]:+.6f}  diff={diff[i]:+.6f}")
    return diff


def main():
    parser = argparse.ArgumentParser(description="Diagnose Hailo INT8 action errors vs CPU FP32")
    parser.add_argument(
        "--cpu-path",
        type=str,
        default="outputs/migrated/act_aloha_sim_transfer_cube_human",
        help="Path to CPU FP32 policy checkpoint",
    )
    parser.add_argument(
        "--hailo-path",
        type=str,
        default="outputs/migrated/act_aloha_sim_transfer_cube_human_hailo_opt2",
        help="Path to Hailo INT8 policy checkpoint",
    )
    args = parser.parse_args()

    print("Loading policies...")
    cpu_policy = load_policy(args.cpu_path)
    hailo_policy = load_policy(args.hailo_path)

    print("Creating environment...")
    env = make_env()

    # Run 5 seeds and collect statistics
    all_action_diffs = []
    all_feature_cosines = []

    for seed in range(42, 47):
        obs, info = env.reset(seed=seed)
        # Raw obs for backbone comparison (no batch dim)
        img_tensor = torch.from_numpy(obs["pixels"]["top"]).permute(2, 0, 1).float() / 255.0

        # Batched obs for select_action (needs batch dim)
        obs_dict = {
            "observation.images.top": img_tensor.unsqueeze(0),  # (1, 3, H, W)
            "observation.state": torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0),  # (1, 14)
        }

        print(f"\n{'=' * 60}")
        print(f"Seed {seed}")
        print(f"{'=' * 60}")

        # Get backbone features
        img = img_tensor.unsqueeze(0)  # (1, 3, H, W)
        cpu_feat = cpu_policy.model.backbone(img)["feature_map"]
        hailo_feat = hailo_policy.model.backbone(img)["feature_map"]
        cosine, feat_diff = compare_features(cpu_feat, hailo_feat, label=f"Seed {seed}")
        all_feature_cosines.append(cosine)

        # Get full actions
        cpu_policy.reset()
        hailo_policy.reset()
        cpu_action = cpu_policy.select_action(obs_dict)
        hailo_action = hailo_policy.select_action(obs_dict)
        action_diff = compare_actions(cpu_action.squeeze(0), hailo_action.squeeze(0), label=f"Seed {seed}")
        all_action_diffs.append(action_diff.numpy())

    # Overall statistics
    all_action_diffs = np.array(all_action_diffs)  # (5, 14)
    print(f"\n{'=' * 60}")
    print("SUMMARY ACROSS 5 SEEDS")
    print(f"{'=' * 60}")
    print(f"Feature cosine: mean={np.mean(all_feature_cosines):.6f}, std={np.std(all_feature_cosines):.6f}")
    print("\nPer-joint action diff statistics:")
    print(f"  {'Joint':>5s}  {'Mean':>10s}  {'Std':>10s}  {'|Mean/Std|':>10s}  {'Verdict':>10s}")
    for j in range(14):
        m = all_action_diffs[:, j].mean()
        s = all_action_diffs[:, j].std()
        ratio = abs(m) / (s + 1e-8)
        verdict = "BIASED" if ratio > 2 else "noisy"
        print(f"  J{j:2d}:   {m:+.6f}  {s:.6f}  {ratio:.3f}       {verdict}")

    print(
        f"\nOverall action diff: mean={all_action_diffs.mean():.6f}, abs_mean={np.abs(all_action_diffs).mean():.6f}"
    )
    print(
        f"Overall bias test: mean/std = {abs(all_action_diffs.mean()) / (all_action_diffs.std() + 1e-8):.3f}"
    )

    # Also check: what is the action SCALE? Compare action magnitudes to diff magnitudes
    print("\n=== ACTION SCALE ANALYSIS ===")
    obs, info = env.reset(seed=42)
    obs_dict = {
        "observation.images.top": torch.from_numpy(obs["pixels"]["top"]).permute(2, 0, 1).float() / 255.0,
        "observation.state": torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0),
    }
    obs_dict["observation.images.top"] = obs_dict["observation.images.top"].unsqueeze(0)
    cpu_policy.reset()
    cpu_action = cpu_policy.select_action(obs_dict).squeeze(0)
    print(f"CPU action magnitudes: {cpu_action.abs().numpy()}")
    print(f"CPU action range: [{cpu_action.min():.6f}, {cpu_action.max():.6f}]")
    print(
        f"Mean action diff / mean action magnitude: {np.abs(all_action_diffs).mean() / cpu_action.abs().mean():.4f}"
    )
    print("  (this is the relative error in normalized action space)")

    env.close()


if __name__ == "__main__":
    main()
