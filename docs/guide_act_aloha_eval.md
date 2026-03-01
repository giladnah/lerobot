# Evaluating ACT Policy in Aloha Simulation

Guide for running pretrained ACT (Action Chunking with Transformers) policy inference in the Aloha dual-arm MuJoCo simulation.

## ACT Architecture Overview

The ACT inference pipeline has 3 components:

1. **Vision Backbone** — ResNet-18 (pretrained on ImageNet) extracts spatial features from camera images
2. **Transformer Encoder** — Fuses image features + robot joint positions + a latent variable (zeros at inference) via self-attention (4 layers, 8 heads, 512-dim)
3. **Transformer Decoder** — Generates a "chunk" of 100 future actions via cross-attention to encoder outputs (1 layer, DETR-style learned queries)

During inference: images go through ResNet, features are fed to the encoder with joint state, the decoder outputs 100 timesteps of actions, actions are executed sequentially from a queue, and the model is re-queried when the queue empties.

Key source files:

| File | Contents |
|------|----------|
| `src/lerobot/policies/act/modeling_act.py` | Full ACT model — `select_action()` is the inference entry point |
| `src/lerobot/policies/act/configuration_act.py` | Hyperparameters (chunk_size, dim_model, n_heads, etc.) |
| `src/lerobot/policies/act/processor_act.py` | Pre/post-processing (normalization, device placement) |
| `src/lerobot/envs/configs.py` | Aloha environment config (obs size, action dim, FPS) |
| `src/lerobot/scripts/lerobot_eval.py` | Eval script that orchestrates everything |
| `src/lerobot/envs/factory.py` | Environment creation factory |

## Prerequisites

### 1. Create virtualenv and install dependencies

```bash
python -m venv .venv
uv sync --extra aloha --extra test
```

This installs `gym-aloha` (MuJoCo-based Aloha sim), `lerobot`, and test dependencies.

### 2. Verify GPU

```bash
.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

The policy runs on GPU; the simulation environments run on CPU. This is normal.

## Checkpoint Migration

The pretrained HuggingFace Hub checkpoints (`lerobot/act_aloha_sim_transfer_cube_human`, `lerobot/act_aloha_sim_insertion_human`) use an old format with normalization baked into model weights. The current codebase expects separate `policy_preprocessor.json` / `policy_postprocessor.json` files.

You **must migrate** before running eval:

```bash
.venv/bin/python src/lerobot/processor/migrate_policy_normalization.py \
  --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
  --output-dir outputs/migrated/act_aloha_sim_transfer_cube_human

.venv/bin/python src/lerobot/processor/migrate_policy_normalization.py \
  --pretrained-path lerobot/act_aloha_sim_insertion_human \
  --output-dir outputs/migrated/act_aloha_sim_insertion_human
```

The migration tool:
1. Downloads the checkpoint from HF Hub
2. Extracts normalization statistics from model weights
3. Removes normalization layers from the model
4. Creates `policy_preprocessor.json` and `policy_postprocessor.json`
5. Saves the clean model + processors to the output directory

## Running Evaluation

### TransferCube (pick up and transfer a cube between arms)

```bash
MUJOCO_GL=egl .venv/bin/lerobot-eval \
  --policy.path=outputs/migrated/act_aloha_sim_transfer_cube_human \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.device=cuda
```

### Insertion (peg insertion task)

```bash
MUJOCO_GL=egl .venv/bin/lerobot-eval \
  --policy.path=outputs/migrated/act_aloha_sim_insertion_human \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.device=cuda
```

### Notes

- `MUJOCO_GL=egl` is required for headless rendering (no display). Omit it if you have a display.
- `--eval.batch_size=1` is recommended for GPUs with limited VRAM (e.g. 2GB MX550). Increase if you have more VRAM — this controls parallel environment count, not model batch size.
- `--policy.device=cpu` works too, just slower.

## Expected Results

From our run on an NVIDIA MX550 (2GB VRAM):

| Task | Success Rate | Avg Sum Reward | Avg Max Reward | Time |
|------|-------------|---------------|---------------|------|
| TransferCube | 70% (7/10) | 191.4 | 3.1 | 50.8s |
| Insertion | 20% (2/10) | 268.3 | 2.4 | 61.2s |

TransferCube is the easier task. Insertion requires precise coordination and has lower success rates. Results are stochastic — numbers will vary across runs.

## Output Files

Evaluation outputs land in `outputs/eval/<date>/<time>_aloha_act/`:
- `videos/aloha_0/eval_episode_N.mp4` — per-episode simulation videos

Play a video:
```bash
xdg-open outputs/eval/<date>/<time>_aloha_act/videos/aloha_0/eval_episode_0.mp4
```
