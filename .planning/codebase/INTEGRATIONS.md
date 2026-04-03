# External Integrations

**Analysis Date:** 2026-04-02

## APIs & External Services

**Model Hosting:**
- HuggingFace Hub — Model weights download (`facebook/tribev2`, `meta-llama/Llama-3.2-3B`)
  - SDK/Client: `huggingface_hub` (hf_hub_download)
  - Auth: `HF_TOKEN` env var (required for gated LLaMA model)
  - Used in: `tribev2/demo_utils.py` TribeModel.from_pretrained()

**Experiment Tracking:**
- Weights & Biases — Training metrics logging
  - SDK/Client: `wandb` (via neuraltrain WandbLoggerConfig)
  - Auth: `WANDB_ENTITY` env var
  - Used in: `tribev2/main.py` TribeExperiment.run()

**Text-to-Speech:**
- Google TTS (gTTS) — Converts text input to audio for processing pipeline
  - SDK/Client: `gtts`
  - Auth: None (public API)
  - Used in: `tribev2/demo_utils.py` TextToEvents.get_events()

## Data Storage

**Databases:**
- None — All data is file-based (NumPy arrays, pandas DataFrames, YAML configs)

**File Storage:**
- `/scratch/scholar/edraymon/.cache/` — HuggingFace model cache, feature cache
- `/scratch/scholar/edraymon/tribev2-eric/` — Project root with results, logs
- `DATAPATH` env var → study datasets (fMRI data)
- `SAVEPATH` env var → experiment outputs and checkpoints

**Caching:**
- exca TaskInfra caching — Feature extraction results cached to disk (mode: "cached")
- HuggingFace model cache — Downloaded models cached locally
- Feature extractors use `infra.keep_in_ram = True` for training performance

## Authentication & Identity

**Auth Provider:**
- HuggingFace Token — Gated model access only
  - Implementation: `HF_TOKEN` env var, must accept LLaMA 3.2-3B license on huggingface.co

## Monitoring & Observability

**Error Tracking:**
- Python logging module — Standard library logging throughout
  - `tribev2/main.py`: LOGGER with `[%(asctime)s %(levelname)s] %(message)s` format
  - `tribev2/demo_utils.py`: logger with `%(levelname)s - %(message)s` format

**Logs:**
- SLURM stdout/stderr → symlinked to `{infra.folder}/log.stdout`, `log.stderr`
- SLURM job logs at `/scratch/scholar/edraymon/tribev2-eric/logs/`

## CI/CD & Deployment

**Hosting:**
- Purdue Scholar GPU cluster (SLURM-managed)
- No cloud deployment — research workstation only

**CI Pipeline:**
- None — no automated testing or deployment pipeline

## Environment Configuration

**Required env vars (set in .env or shell):**
- `HF_TOKEN` — HuggingFace access token
- `HF_HOME` → `/scratch/scholar/edraymon/.cache/huggingface`
- `TORCH_HOME` → `/scratch/scholar/edraymon/.cache/torch`
- `XDG_CACHE_HOME` → `/scratch/scholar/edraymon/.cache`
- `TRANSFORMERS_CACHE` → `/scratch/scholar/edraymon/.cache/huggingface`

**For training (optional):**
- `DATAPATH` — Study datasets directory
- `SAVEPATH` — Experiment output directory
- `SLURM_PARTITION` — Target SLURM partition
- `SLURM_CONSTRAINT` — GPU constraint (e.g., V100, A30)
- `WANDB_ENTITY` — W&B team/user

**Secrets location:**
- `.env` file at `/scratch/scholar/edraymon/tribev2-eric/.env` (gitignored)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Neuroscience Data Sources

**Cortical Meshes:**
- fsaverage5 (~10,242 vertices/hemisphere, ~20,484 total) — Default prediction target
- FreeSurfer subjects dir (`FREESURFER_SUBJECTS_DIR` env var) — Optional for MNI mesh loading
- nilearn datasets — Fetches fsaverage meshes automatically

**Study Datasets:**
- Algonauts2025 / Algonauts2025Bold — `tribev2/studies/algonauts2025.py`
- Lahner2024Bold — `tribev2/studies/lahner2024bold.py`
- Lebel2023Bold — `tribev2/studies/lebel2023bold.py`
- Wen2017 — `tribev2/studies/wen2017.py`

---

*Integration audit: 2026-04-02*
