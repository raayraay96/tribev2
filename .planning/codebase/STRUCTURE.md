# Codebase Structure

**Analysis Date:** 2026-04-02

## Directory Layout

```
tribev2/                         # Project root (git submodule of parent repo)
├── tribev2/                     # Python package — all source code
│   ├── __init__.py              # Package init, exports TribeModel
│   ├── model.py                 # FmriEncoder Transformer (234 lines) — MODIFIABLE
│   ├── main.py                  # TribeExperiment pipeline (651 lines) — reference
│   ├── demo_utils.py            # TribeModel inference API (392 lines)
│   ├── pl_module.py             # PyTorch Lightning module (155 lines)
│   ├── eventstransforms.py      # Custom event transforms (300 lines)
│   ├── utils.py                 # Multi-study loading, splitting (318 lines)
│   ├── utils_fmri.py            # Surface projection, ROI analysis (248 lines)
│   ├── grids/                   # Experiment configurations
│   │   ├── __init__.py
│   │   ├── defaults.py          # Full default hyperparameter config (267 lines)
│   │   ├── configs.py           # Grid search config variants (60 lines)
│   │   ├── run_cortical.py      # Cortical grid search launcher (44 lines)
│   │   ├── run_subcortical.py   # Subcortical grid search launcher (52 lines)
│   │   └── test_run.py          # Quick local test (47 lines)
│   ├── plotting/                # Brain visualization backends
│   │   ├── __init__.py          # Exports plot functions
│   │   ├── base.py              # Base plotting utilities (497 lines)
│   │   ├── cortical.py          # Cortical surface plots - nilearn (311 lines)
│   │   ├── cortical_pv.py       # Cortical surface plots - PyVista (280 lines)
│   │   ├── subcortical.py       # Subcortical volume plots (311 lines)
│   │   └── utils.py             # Plotting helpers (563 lines)
│   └── studies/                 # Dataset definitions
│       ├── __init__.py          # Imports all study classes
│       ├── algonauts2025.py     # Algonauts 2025 competition data (315 lines)
│       ├── lahner2024bold.py    # Lahner 2024 BOLD fMRI data (293 lines)
│       ├── lebel2023bold.py     # Lebel 2023 BOLD fMRI data (344 lines)
│       └── wen2017.py           # Wen 2017 video fMRI data (78 lines)
├── .planning/                   # GSD planning artifacts
├── .agent/                      # GSD Antigravity skills (local install)
├── CLAUDE.md                    # Project context for AI agents (241 lines)
├── README.md                    # Project documentation
├── pyproject.toml               # Package definition + dependencies
├── tribe_demo.ipynb             # Jupyter demo notebook
├── LICENSE                      # CC-BY-NC-4.0
├── CONTRIBUTING.md              # Contribution guidelines
└── CODE_OF_CONDUCT.md           # Code of conduct
```

## Directory Purposes

**`tribev2/` (inner package):**
- Purpose: All Python source code for the TRIBE v2 model
- Contains: Model definition, training pipeline, inference API, data utilities
- Key files: `model.py` (modifiable), `main.py` (reference), `demo_utils.py` (inference)

**`tribev2/grids/`:**
- Purpose: Experiment configuration and grid search launchers
- Contains: Default hyperparameters, config variants, SLURM job launchers
- Key file: `defaults.py` — single source of truth for all hyperparameters

**`tribev2/plotting/`:**
- Purpose: Brain surface visualization (cortical + subcortical)
- Contains: nilearn and PyVista backends, plotting utilities
- Optional dependency: requires `pip install -e ".[plotting]"`

**`tribev2/studies/`:**
- Purpose: Dataset definitions for different neuroscience studies
- Contains: Study-specific data loading, event creation, splitting logic
- Pattern: Each study is a Pydantic model registered via `__init__.py` imports

## Key File Locations

**Entry Points:**
- `tribev2/demo_utils.py`: TribeModel — primary inference interface
- `tribev2/grids/run_cortical.py`: Grid search job submission
- `tribev2/grids/test_run.py`: Local test entry point

**Configuration:**
- `tribev2/grids/defaults.py`: All default hyperparameters and feature extractor configs
- `pyproject.toml`: Package dependencies and metadata

**Core Logic:**
- `tribev2/model.py`: FmriEncoder + FmriEncoderModel — the trainable brain encoding model
- `tribev2/main.py`: Data class + TribeExperiment — full pipeline orchestration
- `tribev2/utils_fmri.py`: TribeSurfaceProjector — cortical mesh projection

**Testing:**
- `tribev2/grids/test_run.py`: Quick smoke test (not pytest)
- No formal test suite exists

## Naming Conventions

**Files:**
- snake_case: `demo_utils.py`, `utils_fmri.py`, `pl_module.py`
- Short descriptive names matching their primary class

**Directories:**
- Lowercase: `grids/`, `plotting/`, `studies/`
- Plural for collections: `studies/` (multiple studies), `grids/` (multiple configs)

**Classes:**
- PascalCase: `FmriEncoder`, `TribeModel`, `TribeExperiment`, `Data`
- Prefix with domain: `FmriTemplateSpace`, `TribeSurfaceProjector`

**Functions:**
- snake_case: `get_events_dataframe()`, `from_pretrained()`, `aggregate_features()`
- Private helpers prefixed with underscore: `_free_extractor_model()`, `_get_checkpoint_path()`

## Where to Add New Code

**New Scoring Method / PFC Analysis:**
- Primary code: `tribev2/utils_fmri.py` (alongside TribeSurfaceProjector)
- Or new file: `tribev2/pfc_scoring.py`

**New Feature Extractor:**
- Implementation: `tribev2/grids/defaults.py` (add config dict)
- neuralset provides the base classes

**New Study Dataset:**
- Implementation: `tribev2/studies/new_study.py`
- Register in: `tribev2/studies/__init__.py`

**New Plotting Backend:**
- Implementation: `tribev2/plotting/new_backend.py`
- Register in: `tribev2/plotting/__init__.py`

**Scaffolding Pipeline Improvements:**
- Located in parent directory: `/scratch/scholar/edraymon/tribev2-eric/scaffolding_pipeline.py`
- This is OUTSIDE the git submodule but imports from `tribev2` package

## Special Directories

**`cache/`:**
- Purpose: HuggingFace model downloads + feature extraction cache
- Generated: Yes (at runtime)
- Committed: No (gitignored)

**`.planning/`:**
- Purpose: GSD planning artifacts
- Generated: Yes (by GSD commands)
- Committed: Yes (per GSD config)

**`tribev2.egg-info/`:**
- Purpose: Editable install metadata
- Generated: Yes (by pip install -e .)
- Committed: No (gitignored)

---

*Structure analysis: 2026-04-02*
