# Coding Conventions

**Analysis Date:** 2026-04-02

## Naming Patterns

**Files:**
- snake_case for all modules: `demo_utils.py`, `utils_fmri.py`, `pl_module.py`
- Study files named after study: `algonauts2025.py`, `lahner2024bold.py`

**Functions:**
- snake_case: `get_events_dataframe()`, `from_pretrained()`, `aggregate_features()`
- Private with underscore prefix: `_free_extractor_model()`, `_get_checkpoint_path()`
- Property decorators for derived values: `@property def device`, `@property def TR`

**Variables:**
- snake_case: `feature_dims`, `n_outputs`, `batch_segments`
- UPPER_CASE for module-level constants: `VALID_SUFFIXES`, `LOGGER`, `N_CPUS`, `CACHEDIR`
- Single-letter for loop/math vars: `x`, `B`, `T`, `H` (batch, time, hidden)

**Types:**
- PascalCase classes: `FmriEncoder`, `TribeModel`, `TextToEvents`
- Pydantic models for all configuration: every config class extends `pydantic.BaseModel` or `BaseModelConfig`
- Union types use `|` syntax (Python 3.10+): `str | None`, `int | None`

## Code Style

**Formatting:**
- black (line-length=88) — configured in `pyproject.toml`
- isort (profile="black") — configured in `pyproject.toml`

**Linting:**
- No explicit linter config found (no flake8, ruff, pylint config)
- Type hints used consistently throughout (PEP 604 union syntax)

**Line Length:**
- 88 characters (black default)

## Import Organization

**Order:**
1. Standard library imports (`import logging`, `import typing as tp`, `from pathlib import Path`)
2. Third-party imports (`import torch`, `import numpy as np`, `import pandas as pd`, `import pydantic`)
3. Meta internal imports (`import neuralset as ns`, `from neuraltrain.models import BaseModelConfig`)
4. Local imports (`from tribev2.main import TribeExperiment`, `from .model import *`)

**Conventions:**
- `typing as tp` alias used throughout (Meta convention)
- `numpy as np`, `pandas as pd` standard aliases
- Star imports for registration: `from .model import *`, `from .studies import *`, `from .utils_fmri import *`
- Star imports in `main.py` explicitly documented with comment: `# register custom models in neuraltrain`

**Path Aliases:**
- None — all imports are relative or absolute package imports

## Error Handling

**Patterns:**
- `ValueError` for invalid configurations:
  ```python
  if len(provided) != 1:
      raise ValueError(f"Exactly one of text_path, audio_path, video_path must be provided")
  ```
- `FileNotFoundError` for missing input files
- `RuntimeError` for state violations (model not loaded)
- `assert` for internal consistency checks (dimension validation in model forward pass)
- `try/except` only for cleanup operations (`_free_extractor_model`)
- `LOGGER.warning()` for non-fatal issues (missing extractors, missing events)

## Logging

**Framework:** Python `logging` module (standard library)

**Patterns:**
- Module-level logger: `LOGGER = logging.getLogger(__name__)` or `logger = logging.getLogger(__name__)`
- INFO level default with custom formatter
- Two format patterns:
  - `main.py`: `[%(asctime)s %(levelname)s] %(message)s` (with timestamp)
  - `demo_utils.py`: `%(levelname)s - %(message)s` (simple)
- f-string logging (not lazy): `LOGGER.info(f"Loading model from {ckpt_path}")`

## Comments

**When to Comment:**
- Meta copyright header on every file (required)
- Module-level docstrings explaining purpose
- Inline comments for non-obvious logic only
- Class docstrings for public API classes (TribeModel, TribeExperiment)

**Docstrings:**
- Google/NumPy hybrid style:
  ```python
  def from_pretrained(cls, checkpoint_dir, ...):
      """Load a trained model from a checkpoint directory or HuggingFace Hub repo.
      
      Parameters
      ----------
      checkpoint_dir:
          Local directory or HuggingFace Hub repo id
      
      Returns
      -------
      TribeModel
          A ready-to-use model instance
      """
  ```

## Function Design

**Size:** Functions range 10-80 lines, most under 40 lines
**Parameters:** Pydantic config objects preferred over many positional args
**Return Values:** Explicit type hints, tuples for multi-return: `-> tuple[np.ndarray, list]`

## Module Design

**Exports:** Star imports in `__init__.py` for registration, explicit imports for API
**Barrel Files:** `tribev2/studies/__init__.py` imports all study classes for registration
**Registration Pattern:** Star imports in `main.py` trigger side effects that register classes with neuralset/neuraltrain

---

*Convention analysis: 2026-04-02*
