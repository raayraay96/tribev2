# Testing Patterns

**Analysis Date:** 2026-04-02

## Test Framework

**Runner:**
- pytest (listed in `pyproject.toml` optional `[test]` dependencies)
- Config: No `pytest.ini`, `pyproject.toml [tool.pytest]`, or `conftest.py` found

**Assertion Library:**
- Python built-in `assert` statements (no pytest fixtures or parametrize found)

**Run Commands:**
```bash
pytest                  # Would run tests (if any existed)
# No watch mode configured
# No coverage configured
```

## Test File Organization

**Location:**
- No dedicated test directory exists
- Only test file: `tribev2/grids/test_run.py` — functional smoke test, not a unit test

**Naming:**
- `test_run.py` follows `test_` prefix convention but is a runnable script, not pytest

**Structure:**
```
tribev2/
└── grids/
    └── test_run.py        # Smoke test — runs a quick local training experiment
```

## Test Structure

**`test_run.py` Pattern (Smoke Test):**
```python
# This is NOT a pytest test — it's a runnable script that exercises the pipeline
from ..main import TribeExperiment

exp = TribeExperiment(**config_overrides)
exp.infra.clear_job()
out = exp.run()
```

**No formal test patterns exist (describe, it, arrange-act-assert, etc.)**

## Mocking

**Framework:** None used

**Patterns:** None — no mocked tests exist

**What to Mock (if tests were added):**
- HuggingFace Hub downloads (slow, requires auth)
- SLURM job submission (cluster-dependent)
- GPU operations (not available on all machines)
- Feature extraction (slow, requires large models)

**What NOT to Mock:**
- Pydantic config validation
- Tensor operations (small test tensors)
- Event DataFrame creation

## Fixtures and Factories

**Test Data:** None configured

**Location:** N/A

## Coverage

**Requirements:** None enforced

**View Coverage:** N/A

## Test Types

**Unit Tests:**
- None exist

**Integration Tests:**
- `tribev2/grids/test_run.py` serves as a full pipeline integration test
- Requires GPU and all dependencies to run

**E2E Tests:**
- Not used

## Common Patterns

**Validation Testing (Implicit via Pydantic):**
```python
# Pydantic models with extra="forbid" catch invalid configs at construction time
class Data(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
```

**Assert-Based Checks (in Production Code):**
```python
assert data.ndim == 3  # B, T, D
assert len(splits) == 1, f"Timeline {timeline_name} has multiple splits"
```

## Test Gaps & Recommendations

**Critical Missing Tests:**
1. FmriEncoder forward pass with synthetic data
2. Config validation edge cases
3. Event DataFrame creation from text/audio/video
4. Surface projection accuracy
5. PFC vertex extraction correctness

---

*Testing analysis: 2026-04-02*
