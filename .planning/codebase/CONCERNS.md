# Codebase Concerns

**Analysis Date:** 2026-04-02

## Tech Debt

**Star Import Registration Pattern:**
- Issue: `tribev2/main.py` uses `from .model import *`, `from .studies import *`, `from .utils_fmri import *` for side-effect registration with neuralset/neuraltrain
- Files: `tribev2/main.py` (lines 39-47)
- Impact: Unclear what's being registered, hard to trace dependencies, IDE autocomplete breaks
- Fix approach: Explicit registration calls instead of star imports

**Dual Logging Formats:**
- Issue: `main.py` uses `[%(asctime)s %(levelname)s]` format while `demo_utils.py` uses `%(levelname)s - %(message)s`
- Files: `tribev2/main.py` (line 52), `tribev2/demo_utils.py` (line 27)
- Impact: Inconsistent log output when both modules are active
- Fix approach: Centralize logging config in `__init__.py`

**f-string Logging:**
- Issue: Uses `LOGGER.info(f"Loading model from {ckpt_path}")` instead of lazy `LOGGER.info("Loading model from %s", ckpt_path)`
- Files: Throughout `tribev2/main.py`, `tribev2/demo_utils.py`
- Impact: String formatting happens even when log level is disabled (minor perf)
- Fix approach: Switch to `%s` style for consistency

## Known Issues

**VRAM Contention Between Feature Extractors:**
- Symptoms: OOM errors when loading multiple large models (LLaMA 3.2-3B + V-JEPA2 ViT-G)
- Files: `tribev2/main.py` `_free_extractor_model()` (lines 59-79)
- Trigger: Running with all three modalities on V100 16GB
- Workaround: `_free_extractor_model()` explicitly deletes models after feature caching, but relies on `gc.collect()` + `torch.cuda.empty_cache()` which is non-deterministic

**Home Quota Exhaustion:**
- Symptoms: Model downloads fail, lock files corrupt cache
- Files: Referenced in `CLAUDE.md` — `.env` file redirects all caches
- Trigger: HuggingFace/PyTorch default to `~/.cache/` which has ~1GB quota on Scholar
- Workaround: All cache env vars redirect to `/scratch/scholar/edraymon/.cache/`

**WhisperX Dependency Chain:**
- Symptoms: Text pipeline fails if WhisperX model downloads hit quota
- Files: `tribev2/eventstransforms.py` (ExtractWordsFromAudio)
- Trigger: First run on clean cache — `faster-whisper-large-v3` download
- Workaround: Pre-download models via `TRANSFORMERS_CACHE` pointing to scratch

## Security Considerations

**HuggingFace Token Exposure:**
- Risk: `HF_TOKEN` in `.env` could leak via logs or error messages
- Files: `.env` (gitignored), `tribev2/demo_utils.py` (uses hf_hub_download)
- Current mitigation: `.env` is gitignored, `.gitignore` covers env files
- Recommendations: Ensure no logging of token values, add `.env` to deny list

**YAML Unsafe Loader:**
- Risk: `yaml.load(f, Loader=yaml.UnsafeLoader)` in `demo_utils.py` line 205 — allows arbitrary Python object instantiation from YAML
- Files: `tribev2/demo_utils.py` (line 205)
- Current mitigation: Config files are self-generated (from_pretrained downloads from HuggingFace)
- Recommendations: Switch to `yaml.SafeLoader` if possible, or validate config source

## Performance Bottlenecks

**Feature Extraction (One-Time):**
- Problem: First run requires extracting features from all stimuli through frozen models
- Files: `tribev2/main.py` Data.get_loaders() → extractor.prepare()
- Cause: LLaMA 3.2-3B, V-JEPA2, Wav2Vec-BERT each process all events
- Improvement path: Already cached after first run. Sequential extraction frees VRAM between models.

**Inference Batch Processing:**
- Problem: `predict()` in `demo_utils.py` iterates batches sequentially with tqdm
- Files: `tribev2/demo_utils.py` (lines 359-380)
- Cause: Single-GPU inference loop, segment-by-segment processing
- Improvement path: Could parallelize across segments with DataParallel for multi-GPU

## Fragile Areas

**FmriEncoder Modality Handling:**
- Files: `tribev2/model.py` `aggregate_features()` (lines 180-225)
- Why fragile: Zero-fills missing modalities with hardcoded dimension calculation (`hidden // len(feature_dims)`). Assumes divisibility.
- Safe modification: Always test with all modality combinations (text-only, audio-only, multimodal)
- Test coverage: None

**Pydantic model_post_init Chains:**
- Files: `tribev2/main.py` (lines 327-376), `tribev2/model.py` (lines 70-76)
- Why fragile: Deep init chains with `super().model_post_init()` — order-dependent, hard to debug
- Safe modification: Add logging before/after each init step
- Test coverage: None

**SubjectLayers Average Mode:**
- Files: `tribev2/main.py` (lines 356-376)
- Why fragile: Multiple conditional branches for average_subjects, complex study name checking
- Safe modification: Only change through config, never directly

## Scaling Limits

**GPU VRAM:**
- Current capacity: V100 16GB / A30 24GB
- Limit: Three frozen extractors + FmriEncoder + optimizer state
- Scaling path: _free_extractor_model() handles sequential extraction, but training with all modalities requires 24GB+

**Cortical Mesh Resolution:**
- Current capacity: fsaverage5 (~20,484 vertices)
- Limit: Higher resolution meshes (fsaverage6: 81,924 vertices) increase prediction head size dramatically
- Scaling path: Low-rank prediction head already implemented (`low_rank_head=2048`)

## Dependencies at Risk

**neuralset==0.0.2 / neuraltrain==0.0.2:**
- Risk: Pinned exact versions of Meta internal packages — no public release cadence
- Impact: Any upstream change breaks compatibility, no semver guarantees
- Migration plan: Fork if needed, but currently stable for research use

**torch>=2.5.1,<2.7:**
- Risk: Uses `torch.compiler` APIs from 2.5+ — breaking changes possible in 2.7
- Impact: Model loading and JIT compilation could break
- Migration plan: Upper bound pinned at <2.7, test before upgrading

## Missing Critical Features

**No Formal Test Suite:**
- Problem: Zero pytest tests, only smoke test in `grids/test_run.py`
- Blocks: Confident refactoring, CI/CD, regression detection

**No CLI Tool:**
- Problem: No command-line interface for inference — requires Python scripting
- Blocks: Easy deployment and integration with scaffolding pipeline

**PFC Vertex Mapping is Approximate:**
- Problem: PFC vertices hardcoded as LH 0-3000, RH 10242-13242 in external scripts
- Blocks: Accurate Brodmann area 9/10/46 targeting
- Note: This is in the parent repo's `scaffolding_pipeline.py`, not in the tribev2 package

## Test Coverage Gaps

**Entire Codebase:**
- What's not tested: Everything — no unit tests exist
- Files: All `.py` files in `tribev2/`
- Risk: Any refactoring could silently break functionality
- Priority: High — FmriEncoder forward pass and config validation should be tested first

---

*Concerns audit: 2026-04-02*
