# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is a Hugging Face library for state-of-the-art machine learning in real-world robotics. It provides policies (ACT, Diffusion, TDMPC, VQ-BeT, SmolVLA, etc.), robot interfaces, dataset management, and simulation environments. Python 3.10+ required.

## Common Commands

### Installation
```bash
uv sync --extra "test"          # Dev install with test deps
uv sync --extra all             # Full install (all policies, robots, extras)
uv sync --extra smolvla         # Specific policy extra
```

### Testing
```bash
pytest -sv ./tests                              # All tests
pytest -sv tests/test_specific_feature.py       # Single test file
pytest -sv tests/test_specific.py::test_func    # Single test function
uv run pytest tests -vv --maxfail=10            # CI-style run
make test-end-to-end DEVICE=cpu                 # End-to-end policy tests
```

Tests require Git LFS artifacts: `git lfs install && git lfs pull`

Set `DEVICE=cpu` or `DEVICE=cuda` env var for hardware selection. Headless rendering: `MUJOCO_GL=egl`.

### Linting & Formatting
```bash
pre-commit install              # One-time setup
pre-commit run --all-files      # Run all checks
```

Tools: **ruff** (format + lint, line-length=110, double quotes), **pyupgrade** (py310+), **typos**, **bandit**, **mypy** (gradual adoption), **prettier** (markdown), **gitleaks**.

### CLI Entry Points
```bash
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human ...
lerobot-eval --policy.path=path/to/checkpoint ...
lerobot-record / lerobot-replay / lerobot-teleoperate / lerobot-calibrate
lerobot-dataset-viz / lerobot-edit-dataset / lerobot-info
```

## Architecture

### Source Layout
- `src/lerobot/policies/` - Policy implementations (each in its own subdir)
- `src/lerobot/robots/` - Robot hardware interfaces
- `src/lerobot/datasets/` - `LeRobotDataset` (Parquet + MP4/images), HF Hub integration
- `src/lerobot/processor/` - Data processing pipeline (normalize, tokenize, device placement)
- `src/lerobot/configs/` - Dataclass configs (draccus-based, supports CLI arg parsing)
- `src/lerobot/scripts/` - CLI entry points
- `src/lerobot/cameras/` - Camera drivers (OpenCV, Intel RealSense)
- `src/lerobot/motors/` - Motor drivers (Dynamixel, FEEtech)
- `src/lerobot/teleoperators/` - Teleoperation interfaces
- `src/lerobot/envs/` - Gym environment integration
- `tests/` - Mirrors source structure

### Key Abstractions

**PreTrainedPolicy** (`policies/pretrained.py`): Base class for all policies. Inherits `nn.Module` + `HubMixin`. Subclasses must set `config_class` and `name` class attributes. Provides HF Hub save/load.

**Robot** (`robots/robot.py`): Abstract base with `connect()`, `disconnect()`, `get_observation()`, `send_action()`. Subclasses set `config_class` and `name`.

**DataProcessorPipeline** (`processor/pipeline.py`): Chains `ProcessorStep` instances. `ProcessorStepRegistry` allows registration by name for serialization. Pipeline is saveable to / loadable from HF Hub.

**Configuration** (draccus): Dataclass-based configs with CLI arg parsing. `TrainPipelineConfig` is the top-level training config containing policy, dataset, and env sub-configs.

### Factory Pattern
`policies/factory.py` uses `get_policy_class(name)` for dynamic imports — avoids loading all policy dependencies at startup. Similar pattern for datasets.

### Registry System
`src/lerobot/__init__.py` maintains `available_policies`, `available_robots`, `available_cameras`, `available_motors`, `available_datasets`, and cross-reference dicts (`available_policies_per_env`, `available_datasets_per_env`).

When adding a new policy: update `available_policies` and `available_policies_per_env` in `__init__.py`, set the `name` class attribute, and update `tests/test_available.py`.

### Dependency Conflicts
`wallx` and `pi` extras pin incompatible `transformers` versions — they conflict with `smolvla`, `groot`, `xvla`, and each other. Managed via `uv` conflict groups in `pyproject.toml`.

## Code Style Notes
- Line length: 110
- Quote style: double
- Docstrings: Google convention
- Ruff rules: E, W, F, I, B, C4, T20, N, UP, SIM
- `__init__.py` files allow unused imports (F401) and star imports (F403)
- mypy strict typing enabled for: `configs`, `envs`, `optim`, `model`, `cameras`, `motors`, `transport`
