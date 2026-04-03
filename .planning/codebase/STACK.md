# Technology Stack

**Analysis Date:** 2026-04-02

## Languages

**Primary:**
- Python 3.11+ — All source code, training scripts, inference pipelines

## Runtime

**Environment:**
- Python 3.11 via Conda on Purdue Scholar GPU cluster (V100 16GB / A30 24GB)
- CUDA 12.1 via `module load cuda/12.1.0`

**Package Manager:**
- pip (via setuptools)
- Conda for environment management
- Lockfile: Not present (dependencies in `pyproject.toml`)

## Frameworks

**Core:**
- PyTorch 2.5.1+cu121 — Deep learning framework, uses `torch.compiler` APIs
- PyTorch Lightning — Training loop abstraction (`tribev2/pl_module.py`)
- Pydantic — Configuration and model validation (used throughout all classes)
- exca (ConfDict, TaskInfra) — Meta's experiment configuration framework

**ML/Neuroscience:**
- neuralset 0.0.2 — Meta's neural dataset library (extractors, dataloaders, segments)
- neuraltrain 0.0.2 — Meta's neural training library (models, losses, metrics, optimizers)
- einops — Tensor rearrangement operations
- x_transformers 1.27.20 — Transformer building blocks

**Feature Extractors (Frozen):**
- LLaMA 3.2-3B (`meta-llama/Llama-3.2-3B`) — Text features via HuggingFace
- Wav2Vec-BERT — Audio features
- V-JEPA2 ViT-G (`facebook/vjepa2-vitg-fpc64-256`) — Video features
- DINOv2-large (`facebook/dinov2-large`) — Image features

**Build/Dev:**
- setuptools >= 61.0 — Build backend
- SLURM — Job scheduler on Scholar cluster

## Key Dependencies

**Critical:**
- `torch>=2.5.1,<2.7` — Core compute, MUST stay 2.5+ for `torch.compiler` APIs
- `neuralset==0.0.2` — Dataset + extractor framework (Meta internal, pinned exact)
- `neuraltrain==0.0.2` — Model + training framework (Meta internal, pinned exact)
- `huggingface_hub` — Model downloads (LLaMA 3.2-3B requires gated access + HF_TOKEN)
- `transformers` — HuggingFace model loading

**Data Processing:**
- `numpy==2.2.6` — Pinned exact for reproducibility
- `pandas` — Event DataFrames
- `pyyaml` — Config file parsing

**Audio/Video Pipeline:**
- `moviepy>=2.2.1` — Video processing
- `gtts` — Text-to-speech synthesis
- `soundfile` — Audio I/O
- `julius` — Audio resampling
- `spacy` — NLP tokenization
- `langdetect` — Language detection for TTS
- `Levenshtein` — String matching for word alignment

**Visualization (optional):**
- `nibabel` — Neuroimaging file formats
- `matplotlib`, `seaborn` — Plotting
- `nilearn` — Brain surface visualization
- `pyvista` — 3D surface rendering
- `colorcet` — Scientific colormaps

## Configuration

**Environment:**
- `HF_TOKEN` — HuggingFace access token (required for LLaMA 3.2-3B gated model)
- `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `TRANSFORMERS_CACHE` — All redirected to `/scratch/scholar/edraymon/.cache/` (home quota ~1GB is too small)
- `DATAPATH` — Path to study datasets
- `SAVEPATH` — Path to experiment outputs
- `SLURM_PARTITION`, `SLURM_CONSTRAINT` — SLURM job configuration
- `WANDB_ENTITY` — Weights & Biases logging
- `.env` file present at parent directory level (gitignored)

**Build:**
- `pyproject.toml` — Package definition with optional dependency groups: `plotting`, `training`, `test`
- No build step needed for inference — `pip install -e .`

## Platform Requirements

**Development:**
- Purdue Scholar cluster login node for submission
- Conda env at `/scratch/scholar/edraymon/tribev2-eric/envs/tribev2`
- SLURM for GPU job submission
- CUDA 12.1 module

**Production/Inference:**
- Single V100 16GB or A30 24GB GPU
- ~15GB model cache on first run (LLaMA 3.2-3B + Wav2Vec-BERT + V-JEPA2)
- 177.2M trainable parameters in FmriEncoder

---

*Stack analysis: 2026-04-02*
