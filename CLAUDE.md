# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kimodo is a **ki**nematic **mo**tion **d**iffusi**o**n model for generating 3D human and robot motions from text prompts and kinematic constraints. It is an NVIDIA research project trained on 700 hours of optical motion capture data. The repo provides inference code, an interactive web demo, a CLI, and a benchmark suite.

## Installation

```bash
pip install -e ".[all]"       # editable install with demo + SOMA extras
pip install -e ".[demo]"      # demo only (viser)
pip install -e ".[soma]"      # SOMA skeleton support only
```

The `MotionCorrection/` C++ extension is built via CMake during `pip install -e .` (set `SKIP_MOTION_CORRECTION_IN_SETUP=1` to skip it). Docker builds use `docker_requirements.txt` with this flag and install MotionCorrection separately.

Requires ~17GB VRAM on GPU. Set `TEXT_ENCODER_DEVICE=cpu` to reduce to <3GB (slower).

## Recommended Workflow

Run three terminals. The text encoder service should stay resident to avoid reloading the large LLM on every generation.

```bash
# Terminal A (persistent): text encoder service
source .venv/bin/activate
export TEXT_ENCODERS_DIR=/home/balance/kimodo/text_encoders
export HF_ENDPOINT=https://hf-mirror.com
kimodo_textencoder                        # full GPU
# TEXT_ENCODER_DEVICE=cpu kimodo_textencoder  # if VRAM < 16 GB

# Terminal B: CLI generation
source .venv/bin/activate
kimodo_gen "A person walks forward." --model Kimodo-SOMA-RP-v1.1 --duration 5 --output output

# Terminal C: interactive demo (http://localhost:7860)
source .venv/bin/activate
export TEXT_ENCODERS_DIR=/home/balance/kimodo/text_encoders
export HF_ENDPOINT=https://hf-mirror.com
kimodo_demo
```

The text encoder listens on `http://0.0.0.0:9550`. All `kimodo_gen` / `kimodo_demo` calls auto-connect to it.

### CLI Examples

```bash
# SOMA human motion (default recommended model)
kimodo_gen "A person walks forward." --model Kimodo-SOMA-RP-v1.1 --duration 5.0 --output output
# add --bvh to also export BVH

# G1 robot motion (outputs .npz + .csv MuJoCo qpos)
kimodo_gen "A person walks forward." --model Kimodo-G1-RP-v1 --duration 5.0 --output g1_walk

# Docker: text encoder + demo
docker compose up
```

## Visualization

| Method | Skeleton | Command |
|---|---|---|
| Interactive Demo | All | `kimodo_demo`, then load `.npz` via Load/Save panel |
| Convert to BVH (Blender etc.) | SOMA | `kimodo_convert output.npz output.bvh --from kimodo --to soma-bvh` |
| MuJoCo viewer | G1 | Edit CSV path in `kimodo/scripts/mujoco_load.py:12`, then `python -m kimodo.scripts.mujoco_load` |

The demo also supports exporting BVH / CSV / AMASS NPZ / video / screenshots via the Exports panel, and adding constraints via the Constraints panel.

## Linting and Formatting

Pre-commit hooks are configured (`.pre-commit-config.yaml`):
- **ruff**: import sorting (`I001`) and formatting (line length 120)
- **docformatter**: sphinx-style docstrings, 100-char wrap
- **prettier**: YAML formatting
- **trailing-whitespace** and **end-of-file-fixer**

```bash
pre-commit run --all-files
```

No test suite exists in the main `kimodo/` package. The `MotionCorrection/` extension has `run_test.py`.

## Architecture

### Inference Pipeline

The core inference flow: `load_model()` -> `Kimodo.__call__()` -> denoising loop -> post-processing.

1. **Model Loading** (`kimodo/model/load_model.py`): `load_model(name)` resolves a short key (e.g. `"soma"`, `"g1"`) or full name (e.g. `"Kimodo-SOMA-RP-v1"`) via the registry, downloads from Hugging Face if needed, loads config.yaml + checkpoint, and assembles the `Kimodo` nn.Module.

2. **Text Encoding** (`kimodo/model/llm2vec.py`, `kimodo/model/text_encoder_api.py`): Uses LLM2Vec (Llama-3-8B fine-tuned) for text embeddings. Can run locally or connect to a remote server (`TEXT_ENCODER_URL`). Mode selection: `TEXT_ENCODER_MODE=auto|local|api`.

3. **Diffusion Model** (`kimodo/model/kimodo_model.py`): The `Kimodo` class wraps a transformer-based denoiser (`TwostageDenoiser` in `backbone.py`) with DDIM sampling (`diffusion.py`) and classifier-free guidance (`cfg.py`). Supports single-prompt and multi-prompt (sequential segment) generation.

4. **Motion Representation** (`kimodo/motion_rep/`): Converts between raw joint rotations/positions and the model's internal feature space. `KimodoMotionRep` handles normalization, feature extraction, and inverse mapping.

5. **Constraints** (`kimodo/constraints.py`): Three constraint types condition generation — `Root2DConstraintSet` (2D paths/waypoints), `FullBodyConstraintSet` (full-body keyframe poses), `EndEffectorConstraintSet` (hand/feet positions/rotations).

6. **Post-processing** (`kimodo/postprocess.py`): Foot-skate cleanup and constraint enforcement applied to generated motion. Uses the C++ `MotionCorrection` extension. Disabled for G1 robot skeleton.

### Skeleton System

`kimodo/skeleton/` defines skeleton hierarchies. Four skeleton types:
- `SOMASkeleton30` — 30-joint internal representation (model trains on this)
- `SOMASkeleton77` — 77-joint SOMA output (external API always returns this for SOMA models)
- `SMPLXSkeleton22` — 22-joint SMPL-X
- `G1Skeleton34` — 34-joint Unitree G1 robot

SOMA models internally use 30 joints but output 77 joints via `output_to_SOMASkeleton77()`.

### Model Registry

`kimodo/model/registry.py` is the single source of truth for model names. It parses Hugging Face repo IDs (e.g. `nvidia/Kimodo-SOMA-RP-v1.1`) into structured `ModelInfo` objects with short keys (e.g. `kimodo-soma-rp-v1.1`). Versionless aliases (e.g. `kimodo-soma-rp`) resolve to the latest version. `resolve_model_name()` handles partial/case-insensitive name resolution.

### Interactive Demo

`kimodo/demo/` is a Viser-based 3D web app with timeline editing, constraint authoring, and real-time motion visualization. Key modules: `app.py` (main demo class), `ui.py` (Gradio panels), `generation.py` (async generation), `state.py` (session state), `queue_manager.py` (request queuing).

### Exports

`kimodo/exports/` handles output format conversion:
- Default NPZ (joint positions + rotations)
- AMASS NPZ (SMPL-X compatible)
- MuJoCo CSV (G1 qpos format)
- BVH (SOMA only)

### Benchmark

`benchmark/` contains the evaluation pipeline: `create_benchmark.py` (build test suite from HF), `generate_eval.py` (run generation), `embed_folder.py` (compute TMR embeddings), `evaluate_folder.py` (compute metrics), `parse_folder.py` (parse results).

## Environment Variables

| Variable | Purpose |
|---|---|
| `TEXT_ENCODER_MODE` | `auto` (default), `local`, or `api` |
| `TEXT_ENCODER_URL` | URL for remote text encoder (default `http://127.0.0.1:9550/`) |
| `TEXT_ENCODER_DEVICE` | Set to `cpu` to run text encoder on CPU (saves VRAM) |
| `CHECKPOINT_DIR` | Local directory for model checkpoints |
| `LOCAL_CACHE` | `true` to prefer local HF cache |

## Contributing

All commits must be signed off (`git commit -s`) per DCO. All files need the SPDX license header:
```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```
