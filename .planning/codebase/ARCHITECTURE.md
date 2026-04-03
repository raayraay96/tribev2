# Architecture

**Analysis Date:** 2026-04-02

## Pattern Overview

**Overall:** Config-Driven Multimodal Experiment Pipeline

**Key Characteristics:**
- Pydantic-based configuration models define every component (data, model, training)
- Feature extractors are frozen pre-trained models; only the FmriEncoder is trainable
- exca TaskInfra handles caching, SLURM job submission, and resource management
- Pipeline: Raw stimuli → Feature extraction → Multimodal fusion → Cortical prediction

## Layers

**Feature Extraction Layer (Frozen):**
- Purpose: Extract learned representations from raw stimuli using pre-trained models
- Location: Configured via `tribev2/grids/defaults.py`, implemented in `neuralset` package
- Contains: LLaMA 3.2-3B (text), Wav2Vec-BERT (audio), V-JEPA2 ViT-G (video)
- Depends on: HuggingFace Hub, CUDA, neuralset extractors
- Used by: Data layer feeds features to FmriEncoder
- **THESE ARE READ-ONLY** — Feature extractors are frozen, only encoder/head are modifiable

**Data Layer:**
- Purpose: Load study datasets, create event DataFrames, build DataLoaders
- Location: `tribev2/main.py` (Data class), `tribev2/studies/`
- Contains: Study loaders, event transforms, segment creation, DataLoader building
- Depends on: neuralset (segments, dataloaders), Feature Extraction Layer
- Used by: TribeExperiment.run()

**Model Layer (Trainable):**
- Purpose: Multimodal transformer that fuses features and predicts cortical activations
- Location: `tribev2/model.py` (FmriEncoder, FmriEncoderModel)
- Contains: Per-modality projector MLPs, combiner, TransformerEncoder, prediction head
- Depends on: Data Layer (batched SegmentData), neuraltrain (base classes)
- Used by: Training layer and inference API

**Training Layer:**
- Purpose: PyTorch Lightning training loop with metrics and checkpointing
- Location: `tribev2/pl_module.py` (BrainModule), `tribev2/main.py` (TribeExperiment)
- Contains: fit/test methods, checkpoint management, W&B logging
- Depends on: Model Layer, PyTorch Lightning, neuraltrain
- Used by: Grid search scripts (`tribev2/grids/`)

**Inference Layer:**
- Purpose: High-level API for running predictions from text/audio/video
- Location: `tribev2/demo_utils.py` (TribeModel)
- Contains: from_pretrained(), get_events_dataframe(), predict()
- Depends on: TribeExperiment (inherits), Model Layer
- Used by: External scripts (scaffolding_pipeline.py, run_inference.py)

## Data Flow

**Inference Flow (Text Input):**

1. `TribeModel.from_pretrained("facebook/tribev2")` — Downloads/loads checkpoint
2. `model.get_events_dataframe(text_path="...")` — Text → gTTS → audio → WhisperX transcription → word-level events DataFrame
3. `model.predict(events)` — Events → feature extraction → segment batching → FmriEncoder forward → cortical vertex predictions
4. Output: `(n_timesteps, 20484)` numpy array of fsaverage5 vertex activations

**Training Flow:**

1. `TribeExperiment(**config)` — Pydantic validation of full experiment config
2. `xp.data.get_events()` — Load study events from all configured datasets
3. Feature extraction — Each modality extractor prepares features (cached to disk)
4. `_free_extractor_model()` — Explicitly frees GPU memory after each extractor
5. `xp.data.get_loaders()` — Creates train/val DataLoaders from segments
6. `xp._setup_trainer()` — Builds FmriEncoder model, Lightning Trainer
7. `xp.fit()` / `xp.test()` — PyTorch Lightning training loop

**FmriEncoder Forward Pass:**

1. `aggregate_features(batch)` — Per-modality projector MLPs → concatenation/stacking
2. Optional: temporal smoothing (Gaussian 1D conv)
3. Optional: temporal dropout during training
4. `transformer_forward(x)` — Combiner MLP → positional embedding → TransformerEncoder
5. Optional: low_rank_head linear projection (1152 → 2048)
6. `predictor(x)` — SubjectLayers head → per-vertex predictions
7. `pooler(x)` — AdaptiveAvgPool1d to target timesteps

**State Management:**
- Config-based: All state is in Pydantic models and YAML config files
- Checkpoints: PyTorch Lightning `.ckpt` files with model state + config
- Caching: exca TaskInfra caches feature extraction results to disk

## Key Abstractions

**BaseExperiment (from neuraltrain):**
- Purpose: Standard experiment lifecycle (setup, run, fit, test)
- Examples: `tribev2/main.py` TribeExperiment inherits from it
- Pattern: Config-driven with Pydantic validation + exca TaskInfra

**BaseModelConfig (from neuraltrain):**
- Purpose: Model configuration that builds nn.Module instances
- Examples: `tribev2/model.py` FmriEncoder, TemporalSmoothing
- Pattern: Pydantic model with `.build()` method returning nn.Module

**SubjectLayers (from neuraltrain):**
- Purpose: Per-subject prediction heads (handles multi-subject training)
- Examples: Used in FmriEncoder.predictor
- Pattern: Subject-specific weights with dropout and averaging support

**MultiStudyLoader:**
- Purpose: Load and merge multiple neuroscience study datasets
- Examples: `tribev2/utils.py`
- Pattern: Named study references resolved to study classes in `tribev2/studies/`

## Entry Points

**Inference:**
- Location: `tribev2/demo_utils.py` TribeModel class
- Triggers: External scripts importing TribeModel
- Responsibilities: Load pretrained model, process inputs, generate predictions

**Training (Grid Search):**
- Location: `tribev2/grids/run_cortical.py`, `tribev2/grids/run_subcortical.py`
- Triggers: `python -m tribev2.grids.run_cortical`
- Responsibilities: Submit SLURM grid search jobs

**Local Test:**
- Location: `tribev2/grids/test_run.py`
- Triggers: `python -m tribev2.grids.test_run`
- Responsibilities: Quick local validation run

## Error Handling

**Strategy:** Assertions + ValueError for invalid configurations, logging for warnings

**Patterns:**
- Pydantic `model_post_init()` validates config consistency
- `assert` statements for dimension checks in model forward pass
- `logging.warning()` for non-fatal issues (missing event types, extractor removal)
- `_free_extractor_model()` uses try/except for graceful GPU cleanup

## Cross-Cutting Concerns

**Logging:** Python logging module with per-module loggers (INFO level default)
**Validation:** Pydantic BaseModel/ConfigDict with `extra="forbid"` for strict config
**Authentication:** HuggingFace Hub token for gated model access
**GPU Memory:** Explicit `_free_extractor_model()` + `gc.collect()` + `torch.cuda.empty_cache()` between extractors

---

*Architecture analysis: 2026-04-02*
