# TRIBE v2 — Cognitive Scaffolding BCI System

> **Owner**: Eric Raymond (edraymon@purdue.edu)
> **Cluster**: Purdue Scholar GPU (V100 16GB / A30 24GB)
> **Model**: Meta TRIBE v2 (`facebook/tribev2`) — 177.2M parameters
> **Purpose**: Closed-loop brain-computer interface backend for ADHD cognitive scaffolding

---

## What This Project Is

This project uses Meta's **TRIBE v2** foundation model to predict prefrontal cortex (PFC) brain activity from naturalistic stimuli (text, audio, video). The predicted PFC activation patterns are transformed into **PFC Stability Scores (0–1)** that drive a real-time closed-loop scaffolding system designed to mimic the pharmacological effects of Guanfacine (alpha-2A agonist) through software interventions.

### The Vision: Guanfacine-Mimicking BCI

The full system architecture has three layers:

1. **Hardware Layer (Passive Sensing)**: Muse 2 EEG headband (AF7/AF8 frontal lobe) + Polar H10 HRV chest strap → BLE → PC. Battery-operated, total galvanic isolation, Non-Significant Risk (NSR) for IRB.

2. **Processing Layer (This Codebase)**: TRIBE v2 predicts cortical surface activation (fsaverage5 mesh, ~20,484 vertices) from stimuli. PFC vertex extraction → variance-based stability scoring → hysteresis state machine (SCAFFOLDING / EDGE / STABLE).

3. **Actuation Layer (Software Interventions)**:
   - **Dynamic Visual Vignetting** — Darkens screen peripherals to reduce visual cortex load
   - **Adaptive Task Pacing** — Increases Inter-Stimulus Interval (ISI) from 1.0s → 1.6s for working memory recovery
   - **Phasic Alerting Cues** — 400Hz tone pip + green highlight 150ms before stimulus to trigger orienting reflex

---

## Project Structure

```
/scratch/scholar/edraymon/tribev2-eric/
├── .env                          # HF_TOKEN and secrets (gitignored)
├── .gitignore
├── program.md                    # Karpathy loop research protocol
├── quickstart.sh                 # One-command setup/run/status/results
├── setup_env.sh                  # Conda environment builder
│
├── scaffolding_pipeline.py       # ★ MAIN — PFC stability scoring pipeline
├── run_scaffold.slurm            # SLURM job for scaffolding pipeline
│
├── run_inference.py              # General-purpose inference runner (text/audio/video)
├── run_tribev2.slurm             # SLURM job for general inference
├── run_experiment.slurm          # SLURM job for autoresearch experiments
├── run_interactive.slurm         # Interactive GPU session
│
├── train_experiment.py           # Autoresearch training loop (agent-modifiable)
│
├── tribev2/                      # ★ Meta's TRIBE v2 source (git submodule)
│   ├── tribev2/
│   │   ├── model.py              # FmriEncoder Transformer — modifiable for research
│   │   ├── demo_utils.py         # TribeModel.from_pretrained(), inference API
│   │   ├── main.py               # TribeExperiment pipeline (read-only reference)
│   │   ├── grids/defaults.py     # Default hyperparameters
│   │   ├── plotting/             # Brain surface visualization
│   │   └── ...
│   ├── pyproject.toml
│   └── tribe_demo.ipynb
│
├── stimuli/                      # Input stimuli files
│   └── sample_text.txt
├── cache/                        # HuggingFace model cache (gitignored)
├── results/                      # General inference results (gitignored)
├── scaffolding_results/          # PFC stability results + SLURM logs
├── autoresearch_results/         # Karpathy loop experiment results
├── logs/                         # SLURM job logs (gitignored)
└── envs/                         # Conda environment (gitignored)
    └── tribev2/                  # Python 3.11 + PyTorch 2.5.1 + CUDA 12.1
```

---

## Environment Setup

```bash
# On Scholar login node:
module load conda/2026.03 cuda/12.1.0
conda activate /scratch/scholar/edraymon/tribev2-eric/envs/tribev2

# Or use the quickstart:
./quickstart.sh setup    # Create env + install everything
./quickstart.sh login    # HuggingFace login (needed for LLaMA 3.2-3B)
```

### Critical Environment Notes

- **All caches must go to `/scratch/scholar`** — home quota is tiny (~1GB). The `.env` and all scripts redirect `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `TRANSFORMERS_CACHE` to `/scratch/scholar/edraymon/.cache/`.
- **LLaMA 3.2-3B access required** — TRIBE v2 uses LLaMA 3.2-3B as its text feature extractor. You must accept the license at https://huggingface.co/meta-llama/Llama-3.2-3B and have `HF_TOKEN` set.
- **PyTorch 2.5.1** is installed with CUDA 12.1 support. The model uses `torch.compiler` APIs from PyTorch 2.5+.
- **SLURM scripts must source modules** in non-login shells: `source /etc/profile.d/00-modulepath.sh` + `source /etc/profile.d/modules.sh` before `module load`.

---

## Key Scripts

### `scaffolding_pipeline.py` — The Main Pipeline

Runs 5 stimulus types through TRIBE v2 and extracts PFC stability scores:

| Stimulus | Purpose |
|---|---|
| `neutral_text` | Baseline — bland news/weather content |
| `cognitive_load` | Mental arithmetic + working memory stress |
| `emotional_arousal` | High-valence emergency scenario |
| `sustained_attention` | Go/no-go rapid visual task |
| `relaxation` | Guided breathing / parasympathetic activation |

**PFC Region Mapping** (fsaverage5, ~10,242 vertices/hemisphere):
- Left PFC: vertices 0–3000
- Right PFC: vertices 10242–13242

**Scoring Methods**:
- `inverse_variance` (default) — `1 / (1 + var * 100)` — lower PFC variance = higher stability
- `activation_ratio` — PFC activation relative to whole-brain
- `combined` — 60% inverse variance + 40% temporal stability

**Hysteresis State Machine**:
- STABLE: score > 0.55
- EDGE: 0.40 ≤ score ≤ 0.55
- SCAFFOLDING: score < 0.40 (activates interventions)
- Scaffold stays ON until sustained recovery above 0.55 for 3 seconds

### `run_inference.py` — General Inference

Supports `--mode text|audio|video`, generates brain surface visualizations, hemisphere analysis, and analysis reports.

### `train_experiment.py` — Autoresearch Loop

The Karpathy loop target: modify architecture parameters (depth, hidden dim, attention, etc.) and run fixed 10-minute budget experiments to maximize `val_pearson`.

---

## Running Jobs

```bash
# Scaffolding pipeline (PFC stability scoring)
sbatch run_scaffold.slurm

# General inference
sbatch run_tribev2.slurm

# Autoresearch experiment
sbatch run_experiment.slurm

# Interactive GPU shell
./quickstart.sh interactive

# Check status
squeue -u edraymon
./quickstart.sh status
```

---

## TRIBE v2 Model Architecture

TRIBE v2 fuses three modality encoders into one Transformer:

```
Text (LLaMA 3.2-3B) ─┐
Audio (Wav2Vec-BERT) ──┤── Projector MLPs ── Combiner ── FmriEncoder Transformer ── Prediction Head ── fsaverage5 vertices
Video (V-JEPA2 ViT-G) ┘
```

- **FmriEncoder**: 8-layer Transformer, hidden=1152, low_rank_head=2048
- **Output**: ~20,484 cortical vertex predictions per timestep
- **Feature extractors are FROZEN** — only the encoder/head are modifiable
- **Inference**: Text → TTS → WhisperX transcription → aligned events → model prediction

---

## Known Issues & Gotchas

1. **LLaMA 3.2-3B Gated Access**: Must accept license on HuggingFace AND have valid `HF_TOKEN`. Error: `403 Client Error... Access to model meta-llama/Llama-3.2-3B is restricted`.

2. **Home Quota Exceeded**: The `~/.cache/huggingface` directory fills home quota instantly. All caches MUST be redirected to `/scratch/scholar/edraymon/.cache/`. Watch for lock files in `~/.cache/`.

3. **WhisperX Dependencies**: TRIBE's text pipeline runs TTS → WhisperX for word-level alignment. `faster-whisper-large-v3` model download can hit quota. Ensure `TRANSFORMERS_CACHE` points to scratch.

4. **Missing Events Warning**: `neuralset.extractors.base` warns about all-zero encodings for missing events — this is non-fatal and expected for text-only inputs (no video events).

5. **PyTorch 2.5+ API**: The model uses `torch.compiler` APIs. Don't downgrade below PyTorch 2.5.

---

## Iteration Targets (Karpathy Loop)

The scaffolding pipeline has several knobs to iterate on:

- **PFC vertex ranges** — Current approximation may not perfectly map to Brodmann areas 9/10/46
- **VARIANCE_SCALE** — Controls sensitivity of inverse_variance scoring
- **Thresholds** — 0.40/0.55 are theoretical; calibrate against ground-truth cognitive states
- **Scoring method** — `combined` may outperform pure `inverse_variance`
- **Stimulus design** — Optimize text content for maximally differentiable PFC responses
- **Temporal smoothing** — May reduce noise in per-timestep scores

---

## Related Architecture: Full BCI System

```
┌─────────────────────────────────────────────────────────────┐
│                    HARDWARE LAYER                           │
│  Muse 2 (EEG: AF7/AF8)  ──BLE──┐                          │
│  Polar H10 (HRV: R-R)   ──BLE──┤── Lab Streaming Layer    │
│  Total galvanic isolation        │   (LSL sync)            │
└──────────────────────────────────┤──────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  PROCESSING LAYER                           │
│  Artifact rejection (±100µV blink filter)                   │
│  TRIBE v2 PFC prediction (THIS CODEBASE)                    │
│  PFC Stability Score (0–1)                                  │
│  Hysteresis state machine (SCAFFOLD/EDGE/STABLE)            │
└─────────────────────────────┬───────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  ACTUATION LAYER                            │
│  A. Visual Vignetting (peripheral darkening)                │
│  B. Adaptive ISI Pacing (1.0s → 1.6s)                      │
│  C. Phasic Alerting (400Hz pip + green highlight)           │
│  User experience: seamless, unobtrusive, closed-loop        │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

| Item | Value |
|---|---|
| Model | `facebook/tribev2` (177.2M params) |
| Python | 3.11 |
| PyTorch | 2.5.1+cu121 |
| GPU | V100 16GB or A30 24GB |
| Mesh | fsaverage5 (~20,484 vertices) |
| PFC Vertices | LH: 0–3000, RH: 10242–13242 |
| Conda Env | `/scratch/scholar/edraymon/tribev2-eric/envs/tribev2` |
| Cache | `/scratch/scholar/edraymon/.cache/` |
| HF Token | In `.env` file |
