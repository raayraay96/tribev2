# Project State — TRIBE v2

## Current Phase
Phase 1: PFC Atlas ROI Module

## Project Reference
See: .planning/PROJECT.md (updated 2026-04-02)

**Core value:** Accurate, neurologically consistent PFC stability scoring from TRIBE v2 predictions
**Current focus:** Phase 1 — Replace hardcoded vertex ranges with atlas-based PFC extraction

## Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1 — PFC Atlas ROI Module | **READY** | Awaiting plan + execution |
| 2 — PFC Stability Scoring | Not started | Depends on Phase 1 |
| 3 — Temporal Smoothing | Not started | Depends on Phase 2 |
| 4 — Hysteresis State Machine | Not started | Depends on Phase 3 |
| 5 — Karpathy Loop Automation | Not started | Depends on Phase 4 |
| 6 — Integration & Validation | Not started | Depends on Phase 5 |

## Context Gathered

### From Research
- PFC vertex ranges must use nilearn atlas (Destrieux/Schaefer), not hardcoded indices
- fMRI prediction has poor test-retest reliability → use per-session baseline calibration
- Three scoring methods need head-to-head comparison
- Temporal smoothing must be asymmetric to preserve rapid stability drops
- State machine requires 3-second sustain hysteresis

### From Codebase Map
- TRIBE v2 model: 177.2M params, fsaverage5 mesh (~20,484 vertices)
- Feature extractors are frozen (LLaMA 3.2-3B, Wav2Vec-BERT, V-JEPA2)
- Sequential extraction required for V100 16GB VRAM
- No test suite exists — need to create tests alongside new modules
- Pydantic-based config throughout — new modules should follow same pattern

## Key Files
- `.planning/PROJECT.md` — Project definition
- `.planning/REQUIREMENTS.md` — v1.0 requirements
- `.planning/ROADMAP.md` — 6-phase roadmap
- `.planning/research/SUMMARY.md` — Research synthesis
- `.planning/codebase/ARCHITECTURE.md` — Current architecture
- `CLAUDE.md` — Legacy project context (pre-GSD)

---
*State updated: 2026-04-02 — project initialized*
