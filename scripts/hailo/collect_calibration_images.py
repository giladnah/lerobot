"""Collect preprocessed calibration images from the Aloha simulation.

Runs random actions in the simulation, captures camera frames, and applies
the same preprocessing (MEAN_STD normalization) that the ACT policy uses.
Saves the result as a numpy array suitable for Hailo HEF compilation.

Usage:
    MUJOCO_GL=egl python scripts/hailo/collect_calibration_images.py \
        --policy-path outputs/migrated/act_aloha_sim_transfer_cube_human \
        --n-frames 1024 \
        --output scripts/hailo/artifacts/calibration_images.npy
"""

import argparse
from pathlib import Path

import gym_aloha  # noqa: F401 — registers gym envs
import gymnasium as gym
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors


def collect_frames(
    policy_path: str, n_frames: int, env_task: str = "gym_aloha/AlohaTransferCube-v0"
) -> np.ndarray:
    """Collect preprocessed camera frames from the Aloha simulation."""
    # Load policy config and preprocessor (for normalization stats)
    config = PreTrainedConfig.from_pretrained(policy_path)
    preprocessor, _ = make_pre_post_processors(config)

    # Create environment
    env = gym.make(env_task, obs_type="pixels_agent_pos")
    obs, _ = env.reset()

    frames = []
    print(f"Collecting {n_frames} preprocessed frames...")

    while len(frames) < n_frames:
        # Take random action
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()
            continue

        # Build observation dict matching policy input format
        obs_dict = {
            "observation.images.top": torch.from_numpy(obs["pixels"]["top"]).permute(2, 0, 1).float() / 255.0,
            "observation.state": torch.from_numpy(obs["agent_pos"]).float(),
        }

        # Run through preprocessor (adds batch dim, moves to device, normalizes)
        processed = preprocessor(obs_dict)

        # Extract the normalized image tensor
        img = processed["observation.images.top"]  # (1, 3, H, W)
        frames.append(img.squeeze(0).cpu().numpy())  # (3, H, W)

        if len(frames) % 100 == 0:
            print(f"  {len(frames)}/{n_frames}")

    env.close()

    # Stack and convert from NCHW to NHWC (Hailo convention)
    frames_nchw = np.stack(frames[:n_frames], axis=0)  # (N, 3, H, W)
    frames_nhwc = np.transpose(frames_nchw, (0, 2, 3, 1))  # (N, H, W, 3)

    print(f"Collected {frames_nhwc.shape[0]} frames, shape: {frames_nhwc.shape}")
    print(f"Value range: [{frames_nhwc.min():.3f}, {frames_nhwc.max():.3f}]")
    print(f"Mean: {frames_nhwc.mean():.3f}, Std: {frames_nhwc.std():.3f}")

    return frames_nhwc.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Collect calibration images from Aloha simulation")
    parser.add_argument(
        "--policy-path",
        type=str,
        default="outputs/migrated/act_aloha_sim_transfer_cube_human",
        help="Path to migrated policy checkpoint",
    )
    parser.add_argument("--n-frames", type=int, default=1024, help="Number of frames to collect")
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/hailo/artifacts/calibration_images.npy",
        help="Output numpy file",
    )
    parser.add_argument(
        "--env-task", type=str, default="gym_aloha/AlohaTransferCube-v0", help="Gym env task ID"
    )
    args = parser.parse_args()

    frames = collect_frames(args.policy_path, args.n_frames, args.env_task)
    np.save(args.output, frames)
    print(f"Saved to {args.output} ({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
